#!/usr/bin/env python
# coding=utf-8
"""
Textual Inversion SDXL Inference Script
This script generates images using trained textual inversion embeddings
"""

import argparse
import torch
from pathlib import Path
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using trained textual inversion embeddings.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--learned_embeds_path",
        type=str,
        required=True,
        help="Path to the learned embeddings file for text_encoder (learned_embeds.safetensors).",
    )
    parser.add_argument(
        "--learned_embeds_2_path",
        type=str,
        required=True,
        help="Path to the second learned embeddings file for text_encoder_2 (learned_embeds_2.safetensors) for SDXL.",
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        required=True,
        help="The placeholder token used during training (e.g., <your-concept>).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="The prompt to use for image generation.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="The negative prompt to use for image generation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_images",
        help="Directory where generated images will be saved.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=4,
        help="Number of images to generate.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for classifier-free guidance.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Height of generated images.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Width of generated images.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="fp16",
        help="Variant of the model files (e.g., fp16).",
    )
    parser.add_argument(
        "--use_safetensors",
        action="store_true",
        default=True,
        help="Whether to use safetensors format.",
    )
    parser.add_argument(
        "--token_1",
        type=str,
        default=None,
        help="Optional: different token name for text_encoder. If not provided, uses placeholder_token.",
    )
    parser.add_argument(
        "--token_2",
        type=str,
        default=None,
        help="Optional: different token name for text_encoder_2. If not provided, uses placeholder_token.",
    )
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("Starting Image Generation")
    print("=" * 50)
    print(f"Pretrained Model: {args.pretrained_model_name_or_path}")
    print(f"Learned Embeddings (text_encoder): {args.learned_embeds_path}")
    print(f"Learned Embeddings (text_encoder_2): {args.learned_embeds_2_path}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Prompt: {args.prompt}")
    print(f"Number of Images: {args.num_images}")
    print("=" * 50)
    print()
    
    # Load the pipeline
    print("Loading SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16,
        variant=args.variant,
        use_safetensors=args.use_safetensors
    ).to("cuda")
    
    # Load the learned embeddings for both text encoders
    print("Loading learned embeddings...")
    
    # Determine token names
    token_1 = args.token_1 if args.token_1 else args.placeholder_token
    token_2 = args.token_2 if args.token_2 else args.placeholder_token
    
    # Load embeddings for text_encoder
    try:
        pipe.load_textual_inversion(
            args.learned_embeds_path, 
            token=token_1,
            text_encoder=pipe.text_encoder,
            tokenizer=pipe.tokenizer,
            dtype=torch.bfloat16
        )
        print(f"✓ Loaded text_encoder embeddings from {Path(args.learned_embeds_path).name} with token '{token_1}'")
    except Exception as e:
        print(f"✗ Error: Could not load text_encoder embeddings: {e}")
        return
    
    # Load embeddings for text_encoder_2
    try:
        pipe.load_textual_inversion(
            args.learned_embeds_2_path,
            token=token_2,
            text_encoder=pipe.text_encoder_2,
            tokenizer=pipe.tokenizer_2,
            dtype=torch.bfloat16
        )
        print(f"✓ Loaded text_encoder_2 embeddings from {Path(args.learned_embeds_2_path).name} with token '{token_2}'")
    except Exception as e:
        print(f"✗ Error: Could not load text_encoder_2 embeddings: {e}")
        return
    
    print(f"\nGenerating {args.num_images} images...")
    print(f"Prompt: {args.prompt}")
    if args.negative_prompt:
        print(f"Negative Prompt: {args.negative_prompt}")
    print()
    
    # Set up generator for reproducibility
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        print(f"Using seed: {args.seed}")
    
    # Generate images
    for i in range(args.num_images):
        print(f"Generating image {i+1}/{args.num_images}...")
        
        # Generate single image
        output = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            generator=generator,
        )
        
        image = output.images[0]
        
        # Save image
        if args.seed is not None:
            output_path = output_dir / f"generated_{i:03d}_seed_{args.seed}.png"
        else:
            output_path = output_dir / f"generated_{i:03d}.png"
        
        image.save(output_path)
        print(f"  ✓ Saved: {output_path}")
    
    print()
    print("=" * 50)
    print("Generation completed!")
    print(f"Images saved to: {args.output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
