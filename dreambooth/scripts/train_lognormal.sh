MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
DATA_ROOT="/root/sangoh/CS492-Visual_Content_Generation/dreambooth/dataset/nupjuki-new_data-color"
# OUTPUT_ROOT="./results/dreambooth-sdxl"
OUTPUT_ROOT="./results-verification/dreambooth-sdxl"

SEED=42
BATCH_SIZE=1
NUM_TRAIN_EPOCHS=500
CKPT_STEP=50
LR=1e-4
GRAD_ACC_STEPS=4

accelerate launch train_text_to_image_lora_sdxl_log_normal.py \
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
  --validation_steps 25 \
  --validation_prompt="a blue sks plush running in the jungle. graphic illustration, comic art, graphic novel art, vibrant, highly detailed" \
  # --report_to="wandb" \