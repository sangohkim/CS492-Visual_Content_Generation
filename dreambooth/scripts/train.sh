MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
DATA_ROOT="/root/sangoh/CS492-Visual_Content_Generation/dataset/nupjuki-cropped"
OUTPUT_ROOT="./results/dreambooth-sdxl"

SEED=42
BATCH_SIZE=1
NUM_TRAIN_EPOCHS=1000
CKPT_STEP=100
LR=1e-4
GRAD_ACC_STEPS=4

accelerate launch train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --caption_column="text" \
  --resolution=1024 \
  --train_data_dir=$DATA_ROOT \
  --train_batch_size=$BATCH_SIZE \
  --num_train_epochs=$NUM_TRAIN_EPOCHS \
  --checkpointing_steps=$CKPT_STEP \
  --gradient_accumulation_steps=$GRAD_ACC_STEPS \
  --learning_rate=$LR \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --seed=$SEED \
  --output_dir="$OUTPUT_ROOT/$(basename $DATA_ROOT)" \
  --validation_epochs 10 \
  --validation_prompt="a sks plush jumping on the moon" \
  --report_to="wandb" \