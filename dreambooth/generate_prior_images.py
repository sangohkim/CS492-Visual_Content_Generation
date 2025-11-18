#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate prior-preservation "class images" for DreamBooth SDXL LoRA training.

Usage:
python generate_prior_images.py \
    --output_dir ./class_images/dog \
    --class_prompt "a photo of a dog" \
    --num_images 200 \
    --seed 42

Optionally:
    --model_name stabilityai/stable-diffusion-xl-base-1.0
    --resolution 1024
    --lora_path path/to/lora_weights
"""

import argparse
import torch
from diffusers import StableDiffusionXLPipeline
from pathlib import Path
from tqdm import tqdm
import os


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Folder to save class images")
    parser.add_argument("--class_prompt", type=str, required=True,
                        help="Class prompt to generate prior images")

    parser.add_argument("--num_images", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument("--model_name", type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--resolution", type=int, default=1024)

    parser.add_argument("--lora_path", type=str, default=None,
                        help="Optional: load LoRA when generating prior images")

    parser.add_argument("--torch_dtype", type=str, default="fp16",
                        choices=["fp16", "fp32", "bf16"])

    return parser.parse_args()


def get_dtype(dtype_str):
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "bf16":
        return torch.bfloat16
    return torch.float32


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dtype = get_dtype(args.torch_dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Loading SDXL model: {args.model_name}")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        use_safetensors=True
    ).to(device)

    # optionally load LoRA for consistency between training/inference
    if args.lora_path is not None:
        print(f"[INFO] Loading LoRA weights from: {args.lora_path}")
        pipe.load_lora_weights(args.lora_path)

    generator = torch.Generator(device=device).manual_seed(args.seed)

    total = args.num_images
    batch = args.batch_size

    print(f"[INFO] Generating {total} prior images...")
    for idx in tqdm(range(0, total, batch)):
        cur_bsz = min(batch, total - idx)

        images = pipe(
            prompt=[args.class_prompt] * cur_bsz,
            height=args.resolution,
            width=args.resolution,
            generator=generator
        ).images

        # save each image
        for i, img in enumerate(images):
            save_path = Path(args.output_dir) / f"class_{idx + i:05d}.png"
            img.save(save_path)

    print(f"[INFO] Done! Class images saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
