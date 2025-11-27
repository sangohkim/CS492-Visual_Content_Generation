#!/bin/bash

# Custom Comic Generation Script
# Edit the parameters below to create your own comic

# Change to the script directory
cd "$(dirname "$0")"

###########################################
# CONFIGURATION - Edit these parameters
###########################################

# Character description
GENERAL_PROMPT="a man with a black suit"

# Scene descriptions (add or remove as needed)
PROMPTS=(
    "wake up in the bed"
    "have breakfast"
    "go to work"
    "work in the office"
    "have lunch"
    "go home"
)

# Style (options: "Comic book", "Anime", "Photographic", "Cinematic", etc.)
STYLE="Comic book"

# Model (options: SDXL, RealVision, Juggernaut, Unstable)
MODEL="SDXL"

# Consistent Self-Attention parameters
ID_LENGTH=4          # Number of reference images (recommended: 3-4)
SA32=0.5            # Attention strength for 32x32 (0-1, higher = more consistent)
SA64=0.5            # Attention strength for 16x16 (0-1, higher = more consistent)

# Generation parameters
NUM_STEPS=50        # Inference steps (more = better quality but slower)
GUIDANCE_SCALE=5.0  # CFG scale (higher = stronger prompt following)
HEIGHT=1024         # Image height
WIDTH=1024          # Image width
SEED=42             # Random seed for reproducibility

# VRAM optimization flags (uncomment if needed)
USE_SEQUENTIAL_OFFLOAD=""        # Uncomment next line for max VRAM savings
# USE_SEQUENTIAL_OFFLOAD="--use_sequential_offload"

USE_ATTENTION_SLICING="--use_attention_slicing"  # Enabled by default

USE_VAE_SLICING=""               # Uncomment next line if needed
# USE_VAE_SLICING="--use_vae_slicing"

# Output directory
OUTPUT_DIR="./outputs/my_custom_comic"

# Font settings (optional)
FONT_PATH="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_SIZE=30

###########################################
# DO NOT EDIT BELOW THIS LINE
###########################################

# Convert array to space-separated string
PROMPTS_STR="${PROMPTS[*]}"

echo "=========================================="
echo "Custom Comic Generation"
echo "=========================================="
echo "Character: $GENERAL_PROMPT"
echo "Number of panels: ${#PROMPTS[@]}"
echo "Style: $STYLE"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Run the comic generation
python Comic_Generation.py \
  --general_prompt "$GENERAL_PROMPT" \
  --prompts $PROMPTS_STR \
  --style "$STYLE" \
  --model "$MODEL" \
  --id_length $ID_LENGTH \
  --sa32 $SA32 \
  --sa64 $SA64 \
  --num_steps $NUM_STEPS \
  --guidance_scale $GUIDANCE_SCALE \
  --height $HEIGHT \
  --width $WIDTH \
  --seed $SEED \
  $USE_SEQUENTIAL_OFFLOAD \
  $USE_ATTENTION_SLICING \
  $USE_VAE_SLICING \
  --output_dir "$OUTPUT_DIR" \
  --font_path "$FONT_PATH" \
  --font_size $FONT_SIZE

echo ""
echo "=========================================="
echo "Generation completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
