"""SageMaker inference handler: SDXL with hot-swappable LoRA.

The key capability to demo here: the base model is loaded once, and the LoRA
is toggled and scaled per request. One endpoint serves both the "before" and
"after" images, which is also how a studio would serve many style adapters
off shared GPU rather than one endpoint per style.

Request body:
    {
      "prompt": "flatgem style, a lantern",
      "use_lora": true,
      "lora_scale": 1.0,
      "seed": 42,
      "steps": 30,
      "guidance_scale": 7.0
    }

Response:
    {"image_b64": "...", "meta": {...}}
"""

import base64
import io
import json
import os

import torch
from diffusers import StableDiffusionXLPipeline

BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
ADAPTER_NAME = "flatgem"


def model_fn(model_dir):
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, use_safetensors=True, variant="fp16"
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        pipe.enable_vae_slicing()

    # Locate the LoRA weights unpacked from model.tar.gz.
    weight_file = None
    for cand in ("pytorch_lora_weights.safetensors", "pytorch_lora_weights.bin"):
        if os.path.exists(os.path.join(model_dir, cand)):
            weight_file = cand
            break

    if weight_file:
        pipe.load_lora_weights(model_dir, weight_name=weight_file, adapter_name=ADAPTER_NAME)
        # Start disabled so the first request can show the "before" state.
        pipe.disable_lora()
        print(f"loaded LoRA: {weight_file}", flush=True)
    else:
        print(f"WARNING: no LoRA weights in {model_dir}; base model only", flush=True)

    return {"pipe": pipe, "has_lora": weight_file is not None}


def input_fn(body, content_type="application/json"):
    if content_type != "application/json":
        raise ValueError(f"unsupported content type: {content_type}")
    return json.loads(body)


def predict_fn(data, ctx):
    pipe = ctx["pipe"]
    has_lora = ctx["has_lora"]

    prompt = data.get("prompt", "a potion")
    negative = data.get("negative_prompt", "photograph, 3d render, gradient, blurry")
    use_lora = bool(data.get("use_lora", False)) and has_lora
    scale = float(data.get("lora_scale", 1.0))
    steps = int(data.get("steps", 30))
    guidance = float(data.get("guidance_scale", 7.0))
    seed = int(data.get("seed", 42))

    if has_lora:
        if use_lora:
            pipe.enable_lora()
            pipe.set_adapters([ADAPTER_NAME], adapter_weights=[scale])
        else:
            pipe.disable_lora()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=device).manual_seed(seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=gen,
        height=1024,
        width=1024,
    ).images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return {
        "image_b64": base64.b64encode(buf.getvalue()).decode("utf-8"),
        "meta": {
            "prompt": prompt, "use_lora": use_lora,
            "lora_scale": scale if use_lora else 0.0,
            "seed": seed, "steps": steps, "guidance_scale": guidance,
        },
    }


def output_fn(pred, accept="application/json"):
    return json.dumps(pred), "application/json"
