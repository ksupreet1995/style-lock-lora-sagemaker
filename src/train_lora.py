"""SDXL LoRA fine-tuning for SageMaker Training Jobs.

Trains LoRA adapters on the UNet attention layers only. Text encoders are
left frozen, which is the standard choice for a *style* LoRA: it is more
stable, trains faster, and avoids damaging the model's language understanding.

SDXL specifics handled here:
  - Two text encoders, concatenated prompt embeds plus pooled embeds.
  - add_time_ids conditioning (original size, crop coords, target size).
  - VAE kept in fp32. The SDXL VAE produces NaNs in fp16.

Expects SageMaker channel layout:
  /opt/ml/input/data/train/*.png + metadata.jsonl
  /opt/ml/model/                 <- LoRA weights written here
"""

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from peft import LoraConfig, get_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


class IconDataset(Dataset):
    def __init__(self, root: str, resolution: int):
        self.root = Path(root)
        meta_path = self.root / "metadata.jsonl"
        self.rows = [json.loads(l) for l in open(meta_path, encoding="utf-8") if l.strip()]
        self.tf = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        img = Image.open(self.root / row["file_name"]).convert("RGB")
        return {"pixel_values": self.tf(img), "text": row["text"]}


def collate(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "text": [b["text"] for b in batch],
    }


def encode_prompts(texts, tokenizers, text_encoders, device):
    """SDXL needs both encoders: concat hidden states + pooled from encoder 2."""
    embeds_list = []
    pooled = None
    for i, (tok, enc) in enumerate(zip(tokenizers, text_encoders)):
        ids = tok(
            texts, padding="max_length", max_length=tok.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)

        out = enc(ids, output_hidden_states=True)
        # Penultimate hidden state is the SDXL convention.
        embeds_list.append(out.hidden_states[-2])
        if i == 1:
            pooled = out[0]

    return torch.concat(embeds_list, dim=-1), pooled


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument(
        "--target-modules", type=str, default="attn",
        choices=["attn", "attn_ff", "attn_ff_proj"],
        help=(
            "Which UNet layers get LoRA adapters. 'attn' (v1) is attention "
            "only, which captures style and texture but leaves global "
            "composition to the base model's prior. 'attn_ff_proj' adds the "
            "feed-forward and resnet projection layers, which is where "
            "global structure lives."
        ),
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "data/train"))
    p.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "out"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"device={device} dtype={dtype} rank={args.rank} steps={args.max_steps}", flush=True)

    tok1 = AutoTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer", use_fast=False)
    tok2 = AutoTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer_2", use_fast=False)

    te1 = CLIPTextModel.from_pretrained(BASE_MODEL, subfolder="text_encoder", torch_dtype=dtype).to(device)
    te2 = CLIPTextModelWithProjection.from_pretrained(BASE_MODEL, subfolder="text_encoder_2", torch_dtype=dtype).to(device)
    te1.requires_grad_(False)
    te2.requires_grad_(False)

    # VAE in fp32: the SDXL VAE NaNs in fp16.
    vae = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder="vae", torch_dtype=torch.float32).to(device)
    vae.requires_grad_(False)

    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet", torch_dtype=dtype).to(device)
    unet.requires_grad_(False)

    # v1 used attention only and learned style but not composition. Base SDXL
    # and the v1 LoRA both generated ~0.67 subject coverage against a 0.146
    # target, so the large-object prior sits outside the attention blocks.
    TARGET_SETS = {
        "attn": ["to_k", "to_q", "to_v", "to_out.0"],
        "attn_ff": ["to_k", "to_q", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
        "attn_ff_proj": [
            "to_k", "to_q", "to_v", "to_out.0",
            "ff.net.0.proj", "ff.net.2",
            "proj_in", "proj_out",
            "conv1", "conv2", "conv_shortcut",
            "time_emb_proj",
        ],
    }
    targets = TARGET_SETS[args.target_modules]
    print(f"LoRA targets ({args.target_modules}): {targets}", flush=True)

    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=targets,
    )
    unet.add_adapter(lora_cfg)

    # LoRA params must be fp32 for stable optimisation.
    params = []
    for prm in unet.parameters():
        if prm.requires_grad:
            prm.data = prm.data.float()
            params.append(prm)
    print(f"trainable LoRA params: {sum(p.numel() for p in params):,}", flush=True)

    unet.enable_gradient_checkpointing()
    try:
        unet.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    sched = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")
    opt = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=1e-4)

    ds = IconDataset(args.train_dir, args.resolution)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=2, drop_last=True)
    print(f"dataset size: {len(ds)}", flush=True)

    # Fixed SDXL micro-conditioning: no cropping, square target.
    add_time_ids = torch.tensor(
        [[args.resolution, args.resolution, 0, 0, args.resolution, args.resolution]],
        dtype=dtype, device=device,
    )

    step = 0
    unet.train()
    epochs = math.ceil(args.max_steps * args.grad_accum / max(1, len(dl)))

    for epoch in range(epochs):
        for batch in dl:
            with torch.no_grad():
                px = batch["pixel_values"].to(device, dtype=torch.float32)
                latents = vae.encode(px).latent_dist.sample() * vae.config.scaling_factor
                latents = latents.to(dtype)

                prompt_embeds, pooled = encode_prompts(
                    batch["text"], [tok1, tok2], [te1, te2], device
                )

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, sched.config.num_train_timesteps, (bsz,), device=device).long()
            noisy = sched.add_noise(latents, noise, timesteps)

            pred = unet(
                noisy, timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": pooled,
                    "time_ids": add_time_ids.repeat(bsz, 1),
                },
            ).sample

            loss = F.mse_loss(pred.float(), noise.float(), reduction="mean")
            (loss / args.grad_accum).backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            if step % 50 == 0:
                print(f"step {step}/{args.max_steps * args.grad_accum} loss={loss.item():.4f}", flush=True)

            step += 1
            if step >= args.max_steps * args.grad_accum:
                break
        if step >= args.max_steps * args.grad_accum:
            break

    out = Path(args.model_dir)
    out.mkdir(parents=True, exist_ok=True)
    lora_sd = get_peft_model_state_dict(unet)
    StableDiffusionXLPipeline.save_lora_weights(str(out), unet_lora_layers=lora_sd)

    with open(out / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Saved LoRA weights to {out}", flush=True)
    for f in out.iterdir():
        print(f"  {f.name}  ({f.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
