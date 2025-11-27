#!/bin/bash

# Comic Generation Script
# This script provides easy-to-use examples for generating comics with StoryDiffusion

# Change to the script directory
cd "$(dirname "$0")"

# Generate timestamp for output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_BASE="./outputs/run_${TIMESTAMP}"

# Example 1: Simple comic generation (4 panels)
# echo "=========================================="
# echo "Example 1: Simple 4-panel comic"
# echo "=========================================="
# python Comic_Generation.py \
#   --general_prompt "a man with a black suit" \
#   --prompts "wake up in the bed" "have breakfast" "go to work" "work in the office" \
#   --output_dir ./outputs/example1_simple \
#   --seed 42

# Example 2: Comic with custom style and parameters
echo ""
echo "=========================================="
echo "Example 2: Custom styled comic"
echo "=========================================="

prompts=(
  "reading a newspaper in a cafe"
  "standing on the building in the city"
  "flying through the sky"
  "landing in a burning city under attack by terrorists"
  "battling against the villains with powerful punches"
  "evacuating civilians to safety"
  "standing victorious as the sun breaks through the clouds, people cheering"
)

python Comic_Generation.py \
  --general_prompt "An ironman" \
  --prompts "${prompts[@]}" \
  --style "Comic book" \
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

# Example 3: High quality with VRAM optimization
# echo ""
# echo "=========================================="
# echo "Example 3: High quality with optimizations"
# echo "=========================================="
# python Comic_Generation.py \
#   --general_prompt "a young witch with a purple hat" \
#   --prompts "casting a spell" "flying on a broom" "reading a magic book" "making a potion" \
#   --style "Anime" \
#   --id_length 3 \
#   --sa32 0.5 \
#   --sa64 0.5 \
#   --num_steps 50 \
#   --guidance_scale 7.0 \
#   --height 1024 \
#   --width 1024 \
#   --use_sequential_offload \
#   --use_attention_slicing \
#   --output_dir "${OUTPUT_BASE}_witch" \
#   --seed 42

echo ""
echo "=========================================="
echo "All examples completed!"
echo "Results saved to: ${OUTPUT_BASE}"
echo "=========================================="
