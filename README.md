# CS492-Visual_Content_Generation

![image](demo/comics_final_3.png)

## How to Reproduce
Tested at 20GB GPU VRAM, 96GB disk space server.
### 1. Python Environment
```
conda env create -f requirements.yaml
```

### 2. SDXL checkpoints
```
hf download stabilityai/stable-diffusion-xl-base-1.0
```

### 3. Training
> Expected time: ~2hrs

Note that checkpoints are already provided at `dreambooth/results`. Also we already configured all arguments related to that. At `scripts/train_lognormal.sh`, `OUTPUT_ROOT` is modified to other path not to overwrite the given checkpoints since this is only for verification. If you want to train it from scratch rather than using given checkpoints, then you can uncomment the original `OUTPUT_ROOT` in `scripts/train_lognormal.sh`. Make sure to train at lest 1450 step since our results are based on 1450-step checkpoint.
```
cd dreambooth
bash scripts/train_lognormal.sh
```

## 4. Inference
> Expected time: ~20min
```
bash run_inference_final.sh
```
