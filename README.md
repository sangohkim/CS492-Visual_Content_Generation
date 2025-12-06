# CS492-Visual_Content_Generation

## How to Reproduce
### 1. Python Environment
```
conda env create -f requirements.yml
```
#### Alternative
```
conda create -n <env_name> python=3.10 -y
conda activate <env_name>
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install diffusers==0.35.2 accelerate==1.11.0 huggingface-hub==0.36.0
```
### 2. SDXL checkpoints
```
hf download stabilityai/stable-diffusion-xl-base-1.0
```

### 3. Training
```
cd dreambooth
bash scripts/train_lognormal.sh
```

## 4. Inference
```
bash run_inference_final.sh
```
