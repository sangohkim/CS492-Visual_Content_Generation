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
SEED=42

echo "Using LoRA checkpoint path: $LORA_PATH"

if [ ! -d "$LORA_PATH" ]; then
    echo "Error: LoRA checkpoint directory not found: $LORA_PATH"
    echo "Please update LORA_PATH in this script or train DreamBooth first."
    exit 1
fi

# 변경 가능
# negative prompt 변경하려면 README.md 참고
prompts=(
  "reading a newspaper"
  "running in the jungle"
  "standing in a cave filled with gold"
  "driving a car filled with gold"
  "lying on a pile of gold"
  "standing on a beach at sunset"
)

# 아래 커맨드는 변경 불가
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
  --seed $SEED