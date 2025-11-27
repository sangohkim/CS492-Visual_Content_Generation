#!/bin/bash

# Comic Generation Script with Textual Inversion Support
# This script provides easy-to-use examples for generating comics with StoryDiffusion and trained embeddings

# Change to the script directory
cd "$(dirname "$0")"

# Generate timestamp for output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_BASE="./outputs_ti/run_${TIMESTAMP}"

# ===========================================
# Configuration - Modify these paths
# ===========================================

# Path to your trained textual inversion embeddings
LEARNED_EMBEDS_1="/root/sangoh/CS492-Visual_Content_Generation/textual-inversion/results/monster_toy-archived/learned_embeds-steps-5000.safetensors"
LEARNED_EMBEDS_2="/root/sangoh/CS492-Visual_Content_Generation/textual-inversion/results/monster_toy-archived/learned_embeds_2-steps-5000.safetensors"
PLACEHOLDER_TOKEN="<monster-toy>"

# Check if embeddings exist
if [ ! -f "$LEARNED_EMBEDS_1" ]; then
    echo "Error: Embeddings file not found: $LEARNED_EMBEDS_1"
    echo "Please update LEARNED_EMBEDS_1 path in this script or train textual inversion first."
    exit 1
fi

if [ ! -f "$LEARNED_EMBEDS_2" ]; then
    echo "Error: Embeddings file not found: $LEARNED_EMBEDS_2"
    echo "Please update LEARNED_EMBEDS_2 path in this script or train textual inversion first."
    exit 1
fi

# ===========================================
# Example 1: Simple comic with TI (4 panels)
# ===========================================
# echo "=========================================="
# echo "Example 1: Simple 4-panel comic with TI"
# echo "=========================================="
# python Comic_Generation_TI.py \
#   --general_prompt "$PLACEHOLDER_TOKEN" \
#   --prompts "wake up in the bed" "have breakfast" "go to work" "work in the office" \
#   --learned_embeds_path "$LEARNED_EMBEDS_1" \
#   --learned_embeds_2_path "$LEARNED_EMBEDS_2" \
#   --placeholder_token "$PLACEHOLDER_TOKEN" \
#   --output_dir ./outputs/example1_simple_ti \
#   --seed 42

# ===========================================
# Example 2: Action-packed comic with TI
# ===========================================
echo ""
echo "=========================================="
echo "Example 2: Action comic with TI"
echo "=========================================="

prompts=(
  "standing on a mountain peak, holding a glowing sword"
  "sensing danger, lightning crackling in the sky"
  "jumping off the cliff with determination"
  "landing in a dark forest filled with monsters"
  "battling monsters with swift sword strikes"
  "protecting a village, standing guard at the gate"
  "unleashing a powerful attack, defeating the monster leader"
  "standing victorious as villagers celebrate"
)

python Comic_Generation_TI.py \
  --general_prompt "A photo of $PLACEHOLDER_TOKEN" \
  --prompts "${prompts[@]}" \
  --learned_embeds_path "$LEARNED_EMBEDS_1" \
  --learned_embeds_2_path "$LEARNED_EMBEDS_2" \
  --placeholder_token "$PLACEHOLDER_TOKEN" \
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

# ===========================================
# Example 3: Daily life comic with TI
# ===========================================
# echo ""
# echo "=========================================="
# echo "Example 3: Daily life comic with TI"
# echo "=========================================="
# python Comic_Generation_TI.py \
#   --general_prompt "$PLACEHOLDER_TOKEN, casual clothing" \
#   --prompts "drinking morning coffee" "walking in the park" "reading a book on a bench" "watching the sunset" \
#   --learned_embeds_path "$LEARNED_EMBEDS_1" \
#   --learned_embeds_2_path "$LEARNED_EMBEDS_2" \
#   --placeholder_token "$PLACEHOLDER_TOKEN" \
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
# Example 4: Custom style comic with TI
# ===========================================
# echo ""
# echo "=========================================="
# echo "Example 4: Anime style comic with TI"
# echo "=========================================="
# python Comic_Generation_TI.py \
#   --general_prompt "$PLACEHOLDER_TOKEN, anime style, vibrant colors" \
#   --prompts "training with a wooden sword" "meditating under a waterfall" "sparring with a rival" "achieving inner peace" \
#   --learned_embeds_path "$LEARNED_EMBEDS_1" \
#   --learned_embeds_2_path "$LEARNED_EMBEDS_2" \
#   --placeholder_token "$PLACEHOLDER_TOKEN" \
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
