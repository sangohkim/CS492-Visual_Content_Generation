#!/bin/bash

# Comic Generation Script with DreamBooth LoRA
# This script provides easy-to-use examples for generating comics with StoryDiffusion and DreamBooth LoRA

# Change to the script directory
cd "$(dirname "$0")"

# Generate timestamp for output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_BASE="./outputs_db/run_${TIMESTAMP}"

# ===========================================
# Configuration - Modify these paths
# ===========================================

# Path to your DreamBooth LoRA checkpoint directory
LORA_PATH=$1

echo "Using LoRA checkpoint path: $LORA_PATH"

if [ ! -d "$LORA_PATH" ]; then
    echo "Error: LoRA checkpoint directory not found: $LORA_PATH"
    echo "Please update LORA_PATH in this script or train DreamBooth first."
    exit 1
fi

# ===========================================
# Example 1: Simple comic with DreamBooth LoRA (4 panels)
# ===========================================
# echo "=========================================="
# echo "Example 1: Simple 4-panel comic with DreamBooth LoRA"
# echo "=========================================="
# python Comic_Generation_dreambooth.py \
#   --general_prompt "A sks monster, cute toy style" \
#   --prompts "running in the jungle" "wearing a hat" "jumping on the moon" "playing with a ball in the park" \
#   --lora_path "$LORA_PATH" \
#   --output_dir ./outputs/example1_simple_dreambooth \
#   --seed 42

# ===========================================
# Example 2: Action comic with DreamBooth LoRA
# ===========================================
echo ""
echo "=========================================="
echo "Example 2: Action comic with DreamBooth LoRA"
echo "=========================================="

prompts=(
  "reading a newspaper"
  "running in the jungle"
  "standing in a cave filled with gold"
  "driving a car filled with gold"
  "lying on a pile of gold"
  "standing on a beach at sunset"
)

python Comic_Generation_dreambooth.py \
  --general_prompt "a blue sks plush" \
  --prompts "${prompts[@]}" \
  --lora_path "$LORA_PATH" \
  --style "Comic book storydiffv2" \
  --id_length 4 \
  --sa32 0.75 \
  --sa64 0.75 \
  --num_steps 50 \
  --guidance_scale 5.0 \
  --height 1024 \
  --width 1024 \
  --output_dir "${OUTPUT_BASE}" \
  --use_sequential_offload \
  --use_attention_slicing \
  --seed 42

# ===========================================
# Example 3: Daily life comic with DreamBooth LoRA
# ===========================================
# echo ""
# echo "=========================================="
# echo "Example 3: Daily life comic with DreamBooth LoRA"
# echo "=========================================="
# python Comic_Generation_dreambooth.py \
#   --general_prompt "A sks monster, casual clothing" \
#   --prompts "drinking morning coffee" "walking in the park" "reading a book on a bench" "watching the sunset" \
#   --lora_path "$LORA_PATH" \
#   --style "Photographic" \
#   --id_length 3 \
#   --sa32 0.5 \
#   --sa64 0.5 \
#   --num_steps 50 \
#   --guidance_scale 7.0 \
#   --height 1024 \
#   --width 1024 \
#   --use_sequential_offload \
#   --use_attention_slicing \
#   --output_dir "${OUTPUT_BASE}_daily" \
#   --seed 42

# ===========================================
# Example 4: Anime style comic with DreamBooth LoRA
# ===========================================
# echo ""
# echo "=========================================="
# echo "Example 4: Anime style comic with DreamBooth LoRA"
# echo "=========================================="
# python Comic_Generation_dreambooth.py \
#   --general_prompt "A sks monster, anime style, vibrant colors" \
#   --prompts "training with a wooden sword" "meditating under a waterfall" "sparring with a rival" "achieving inner peace" \
#   --lora_path "$LORA_PATH" \
#   --style "Anime" \
#   --id_length 3 \
#   --sa32 0.6 \
#   --sa64 0.6 \
#   --num_steps 50 \
#   --guidance_scale 6.0 \
#   --height 1024 \
#   --width 1024 \
#   --use_sequential_offload \
#   --use_attention_slicing \
#   --use_vae_slicing \
#   --output_dir "${OUTPUT_BASE}_anime" \
#   --seed 42

echo ""
echo "=========================================="
echo "All examples completed!"
echo "Results saved to: ${OUTPUT_BASE}"
echo "=========================================="
