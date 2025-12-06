#!/usr/bin/env python
# coding=utf-8
"""
Add captions to existing images.
This script takes images and adds caption text to each image, saving them individually.
"""

import argparse
import os
from pathlib import Path
from PIL import Image, ImageFont
from utils.utils import add_caption


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add captions to existing images"
    )
    
    # Input settings
    parser.add_argument(
        "--image_paths",
        type=str,
        nargs='+',
        required=True,
        help="Paths to input images (in order)"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs='+',
        required=True,
        help="Captions for each image (must match number of images)"
    )
    
    # Output settings
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./captioned_images",
        help="Directory to save images with captions"
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="captioned",
        help="Prefix for output filenames"
    )
    
    # Font settings
    parser.add_argument(
        "--font_path",
        type=str,
        default="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        help="Path to font file for caption text"
    )
    parser.add_argument(
        "--font_size",
        type=int,
        default=30,
        help="Font size for captions"
    )
    
    # Caption style settings
    parser.add_argument(
        "--position",
        type=str,
        default="bottom-mid",
        choices=["bottom-mid", "bottom-left", "bottom-right"],
        help="Position of caption on image"
    )
    parser.add_argument(
        "--text_color",
        type=str,
        default="black",
        help="Color of caption text"
    )
    parser.add_argument(
        "--bg_opacity",
        type=int,
        default=200,
        help="Background opacity (0-255)"
    )
    
    args = parser.parse_args()
    return args


def load_images(image_paths):
    """Load images from file paths"""
    images = []
    for path in image_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        img = Image.open(path)
        images.append(img)
    return images


def main():
    args = parse_args()
    
    print(f"Image paths: {args.image_paths}")
    print(f"Prompts: {args.prompts}")
    
    # Validate inputs
    if len(args.image_paths) != len(args.prompts):
        raise ValueError(
            f"Number of images ({len(args.image_paths)}) must match "
            f"number of prompts ({len(args.prompts)})"
        )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Adding Captions to Images")
    print("=" * 80)
    print(f"Number of images: {len(args.image_paths)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Caption position: {args.position}")
    print("=" * 80)
    
    # Load images
    print("\n[1/3] Loading images...")
    images = load_images(args.image_paths)
    print(f"✓ Loaded {len(images)} images")
    
    # Load font
    print("\n[2/3] Loading font...")
    try:
        font = ImageFont.truetype(args.font_path, args.font_size)
        print(f"✓ Loaded font: {args.font_path}")
    except Exception as e:
        print(f"⚠ Could not load font from {args.font_path}: {e}")
        print("  Using default font")
        font = ImageFont.load_default()
    
    # Add captions and save
    print("\n[3/3] Adding captions and saving images...")
    for idx, (image, caption) in enumerate(zip(images, args.prompts)):
        # Add caption to image
        captioned_image = add_caption(
            image,
            text=caption,
            position=args.position,
            font=font,
            text_color=args.text_color,
            bg_opacity=args.bg_opacity
        )
        
        # Generate output filename
        output_path = os.path.join(
            args.output_dir,
            f"{args.output_prefix}_{idx:02d}.png"
        )
        
        # Save image
        captioned_image.save(output_path)
        print(f"✓ Saved: {output_path}")
    
    print("\n" + "=" * 80)
    print("Caption addition completed successfully!")
    print(f"Total images processed: {len(images)}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
