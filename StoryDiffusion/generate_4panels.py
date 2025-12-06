#!/usr/bin/env python
# coding=utf-8
"""
Generate comic layouts from existing images and prompts.
This script takes pre-generated images and creates comic panel layouts with captions.
"""

import argparse
import os
from pathlib import Path
from PIL import Image, ImageFont
from utils.utils import get_comic_4panel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate comic layouts from existing images"
    )
    
    # Input settings
    parser.add_argument(
        "--image_paths",
        type=str,
        nargs='+',
        required=True,
        help="Paths to input images for comic panels (in order)"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs='+',
        required=True,
        help="Captions for each panel (must match number of images)"
    )
    
    # Output settings
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./comic_layouts",
        help="Directory to save generated comic layouts"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="comic",
        help="Base name for output files"
    )
    
    # Font settings
    parser.add_argument(
        "--font_path",
        type=str,
        default="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        help="Path to font file for comic text"
    )
    parser.add_argument(
        "--font_size",
        type=int,
        default=30,
        help="Font size for comic captions"
    )
    
    # Layout settings
    parser.add_argument(
        "--pad_image_path",
        type=str,
        default="/root/sangoh/CS492-Visual_Content_Generation/StoryDiffusion/images/pad_images.png",
        help="Path to padding image for comic layout"
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

    print(f"image_paths: {args.image_paths}")
    print(f"prompts: {args.prompts}")
    
    # Validate inputs
    if len(args.image_paths) != len(args.prompts):
        raise ValueError(
            f"Number of images ({len(args.image_paths)}) must match "
            f"number of prompts ({len(args.prompts)})"
        )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Comic Layout Generation from Existing Images")
    print("=" * 80)
    print(f"Number of panels: {len(args.image_paths)}")
    print(f"Output directory: {args.output_dir}")
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
    
    # Load pad image if exists
    pad_image = None
    if os.path.exists(args.pad_image_path):
        pad_image = Image.open(args.pad_image_path)
        print(f"✓ Loaded padding image: {args.pad_image_path}")
    else:
        print(f"⚠ Padding image not found: {args.pad_image_path}")
        print("  Generating comic without padding")
    
    # Generate comic layout
    print("\n[3/3] Creating comic layout...")
    comics = get_comic_4panel(
        images,
        captions=args.prompts,
        font=font,
        pad_image=pad_image
    )
    
    # Save comics
    print(f"\nSaving comic layouts to {args.output_dir}...")
    for idx, comic in enumerate(comics):
        output_path = os.path.join(
            args.output_dir,
            f"{args.output_name}_page_{idx:02d}.png"
        )
        comic.save(output_path)
        print(f"✓ Saved: {output_path}")
    
    print("\n" + "=" * 80)
    print("Comic layout generation completed successfully!")
    print(f"Total pages created: {len(comics)}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
