#!/bin/bash

# Textual Inversion SDXL Inference Script
# This script generates images using trained textual inversion embeddings

# Configuration
PRETRAINED_MODEL="stabilityai/stable-diffusion-xl-base-1.0"
LEARNED_EMBEDS_PATH="./results/monster_toy-archived/learned_embeds-steps-5000.safetensors"
LEARNED_EMBEDS_2_PATH="./results/monster_toy-archived/learned_embeds_2-steps-5000.safetensors"
OUTPUT_DIR="./generated_images"
PLACEHOLDER_TOKEN="<monster-toy>"

# Generation parameters
PROMPT="A $PLACEHOLDER_TOKEN playing in a futuristic city, highly detailed, vibrant colors, comics style"
NEGATIVE_PROMPT=""
NUM_IMAGES=5
NUM_INFERENCE_STEPS=25
GUIDANCE_SCALE=7.5
HEIGHT=1024
WIDTH=1024
SEED=42

echo "=========================================="
echo "Textual Inversion SDXL Inference"
echo "=========================================="
echo "Model: $PRETRAINED_MODEL"
echo "Text Encoder Embeddings: $LEARNED_EMBEDS_PATH"
echo "Text Encoder 2 Embeddings: $LEARNED_EMBEDS_2_PATH"
echo "Placeholder Token: $PLACEHOLDER_TOKEN"
echo "Prompt: $PROMPT"
echo "Output Directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Run inference using the Python script
python inference.py \
  --pretrained_model_name_or_path="$PRETRAINED_MODEL" \
  --learned_embeds_path="$LEARNED_EMBEDS_PATH" \
  --learned_embeds_2_path="$LEARNED_EMBEDS_2_PATH" \
  --placeholder_token="$PLACEHOLDER_TOKEN" \
  --prompt="$PROMPT" \
  --negative_prompt="$NEGATIVE_PROMPT" \
  --output_dir="$OUTPUT_DIR" \
  --num_images=$NUM_IMAGES \
  --num_inference_steps=$NUM_INFERENCE_STEPS \
  --guidance_scale=$GUIDANCE_SCALE \
  --height=$HEIGHT \
  --width=$WIDTH \
  --seed=$SEED \
  --variant="fp16" \
  --use_safetensors \
  --batch_size 4

echo ""
echo "=========================================="
echo "Done! Check the generated images in: $OUTPUT_DIR"
echo "=========================================="
