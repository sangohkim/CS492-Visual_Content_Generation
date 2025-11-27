# CS492-Visual_Content_Generation

## Installation
### Python Environment
> 잘 안되면 requirements.yaml 참고
```
conda env create -f requirements.yml
```

혹시 위 방법이 안되면 제게 말씀주시고 아래로 진행해주세요.
혹시 아래 방법으로 진행하시는 경우에 지금 나와있는 거 외에 추가로 설치해야하는거 발견하시면 톡방에 말해주세요!

```
conda create -n <env_name> python=3.10 -y
conda activate <env_name>
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install diffusers==0.35.2 accelerate==1.11.0 huggingface-hub==0.36.0
```
### SDXL checkpoints
```
hf download stabilityai/stable-diffusion-xl-base-1.0
```

## Inference
### DreamBooth
> 프롬프트 등 자세한 사항은 `dreambooth/inference_comics_single.py` 참고
```
cd ./dreambooth
python inference_comics_single.py
```

### DreamBooth + StoryDiffusion
> 프롬프트 등 자세한 사항은 `StoryDiffusion/run_comic_generation_dreambooth.sh` 참고
```
cd ./StoryDiffusion
bash run_comic_generation_dreambooth.sh /root/sangoh/CS492-Visual_Content_Generation/dreambooth/results/dreambooth-sdxl/nupjuki-new_data-color-lognormal/checkpoint-900
```

- StoryDiffusion 프롬프트 생성 방식
  - `StoryDiffusion/utils/style_template.py` 에 아래와 같은 형식으로 지정되어 있음. `run_comic_generation_dreambooth.sh` 에서 지정한 prompts 배열 내부의 각 프롬프트가 `{prompt}` 에 매핑되어 최종 프롬프트가 생성됨
  ```
  {
    "name": "Comic book storydiffv2",
    "prompt": "{prompt}. graphic illustration, comic art, graphic novel art, vibrant, highly detailed",
    "negative_prompt": "photograph, deformed, glitch, noisy, realistic, stock photo, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
  }
  ```
  - 변경하고 싶은 경우 위 형식을 복사해서 새로운 `name` 으로 추가하면 됨

