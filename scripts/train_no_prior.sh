#!/bin/bash

# ===== Prior Preservation 없는 기본 학습 스크립트 =====
# Instance images만 사용

# 모델 설정
MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
VAE_NAME="madebyollin/sdxl-vae-fp16-fix"

# 데이터 경로
INSTANCE_DIR="/root/sangoh/CS492-Visual_Content_Generation/dataset/monster_toy"
OUTPUT_ROOT="./results/dreambooth-lora-sdxl"

# 학습 하이퍼파라미터
SEED=42
BATCH_SIZE=1
NUM_TRAIN_EPOCHS=1000
CKPT_STEP=100
LR=5e-5

# 검증 설정
VALIDATION_PROMPT="A sks monster jumping on the moon"
VALIDATION_EPOCHS=10

accelerate launch train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --caption_column="text" \
  --resolution=1024 \
  --train_data_dir=$INSTANCE_DIR \
  --train_batch_size=$BATCH_SIZE \
  --num_train_epochs=$NUM_TRAIN_EPOCHS \
  --checkpointing_steps=$CKPT_STEP \
  --learning_rate=$LR \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --seed=$SEED \
  --output_dir="$OUTPUT_ROOT/$(basename $INSTANCE_DIR)" \
  --validation_epochs=$VALIDATION_EPOCHS \
  --validation_prompt="$VALIDATION_PROMPT" \
  --report_to="wandb"
