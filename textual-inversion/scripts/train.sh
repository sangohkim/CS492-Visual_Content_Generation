#!/bin/bash

# Textual Inversion SDXL Training Script
# This script trains a textual inversion model for SDXL

# Configuration
PRETRAINED_MODEL="stabilityai/stable-diffusion-xl-base-1.0"
TRAIN_DATA_DIR="./dataset/monster_toy/train"
OUTPUT_DIR="./results/monster_toy"
PLACEHOLDER_TOKEN="<monster-toy>"
INITIALIZER_TOKEN="toy"
LEARNABLE_PROPERTY="object"  # or "style"

# Training hyperparameters
RESOLUTION=1024
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
MAX_TRAIN_STEPS=3000
LEARNING_RATE=1e-4
LR_SCHEDULER="constant"
LR_WARMUP_STEPS=0
NUM_VECTORS=1
SAVE_STEPS=500
CHECKPOINTING_STEPS=500
CHECKPOINTS_TOTAL_LIMIT=2
REPEATS=100

# Advanced options (uncomment to use)
# REVISION="main"
# VARIANT="fp16"
# SEED=42
# NUM_TRAIN_EPOCHS=100
# DATALOADER_NUM_WORKERS=0
# USE_8BIT_ADAM=false
# ADAM_BETA1=0.9
# ADAM_BETA2=0.999
# ADAM_WEIGHT_DECAY=1e-2
# ADAM_EPSILON=1e-08
# LR_NUM_CYCLES=1

# Validation settings
VALIDATION_PROMPT="A photo of $PLACEHOLDER_TOKEN"
NUM_VALIDATION_IMAGES=4
VALIDATION_STEPS=5

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Starting Textual Inversion Training"
echo "=========================================="
echo "Pretrained Model: $PRETRAINED_MODEL"
echo "Training Data: $TRAIN_DATA_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Placeholder Token: $PLACEHOLDER_TOKEN"
echo "Initializer Token: $INITIALIZER_TOKEN"
echo "Max Train Steps: $MAX_TRAIN_STEPS"
echo "=========================================="
echo ""

# Run training with accelerate
accelerate launch textual_inversion_sdxl.py \
  --pretrained_model_name_or_path="$PRETRAINED_MODEL" \
  --train_data_dir="$TRAIN_DATA_DIR" \
  --learnable_property="$LEARNABLE_PROPERTY" \
  --placeholder_token="$PLACEHOLDER_TOKEN" \
  --initializer_token="$INITIALIZER_TOKEN" \
  --resolution=$RESOLUTION \
  --train_batch_size=$TRAIN_BATCH_SIZE \
  --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
  --max_train_steps=$MAX_TRAIN_STEPS \
  --learning_rate=$LEARNING_RATE \
  --scale_lr \
  --lr_scheduler="$LR_SCHEDULER" \
  --lr_warmup_steps=$LR_WARMUP_STEPS \
  --num_vectors=$NUM_VECTORS \
  --repeats=$REPEATS \
  --save_steps=$SAVE_STEPS \
  --output_dir="$OUTPUT_DIR" \
  --checkpointing_steps=$CHECKPOINTING_STEPS \
  --checkpoints_total_limit=$CHECKPOINTS_TOTAL_LIMIT \
  --mixed_precision="fp16" \
  --validation_prompt="$VALIDATION_PROMPT" \
  --num_validation_images=$NUM_VALIDATION_IMAGES \
  --validation_steps=$VALIDATION_STEPS \
  --seed=42 \
  # --report_to="wandb" \

# Optional flags (uncomment to use):
# --center_crop \
# --gradient_checkpointing \
# --enable_xformers_memory_efficient_attention \
# --allow_tf32 \
# --use_8bit_adam \
# --save_as_full_pipeline \
# --push_to_hub \
# --seed=$SEED \
# --revision="$REVISION" \
# --variant="$VARIANT" \

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo "Learned embeddings saved to:"
echo "  - $OUTPUT_DIR/learned_embeds.safetensors"
echo "  - $OUTPUT_DIR/learned_embeds_2.safetensors"
echo ""
echo "TensorBoard logs saved to: $OUTPUT_DIR/logs"
echo ""
echo "To view training progress:"
echo "  tensorboard --logdir=$OUTPUT_DIR/logs"
echo ""
echo "To resume training from a checkpoint:"
echo "  Add --resume_from_checkpoint=checkpoint-<step>"
echo "  or --resume_from_checkpoint=latest"
echo "=========================================="
