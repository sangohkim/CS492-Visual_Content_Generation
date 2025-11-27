from huggingface_hub.repocard import RepoCard
from diffusers import DiffusionPipeline
import torch
from pathlib import Path
import re
import os
from datetime import datetime
import yaml

### 변경 불가
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
num_inference_steps = 50
seed = 42

### 변경 가능

# "a blue sks plush"는 되도록 붙어있는게 좋음
# "a blue ~~~ sks plush"나 "a blue sks ~~~ plush"로 해보는것도 좋긴한데 appearance가 좀 달라질 가능성 높음
# "a blue sks plush" 앞뒤로 단어 또는 문장을 추가하는 것이 좋음
# 개수 마음대로 조절 가능.
prompts = [
    "a blue sks plush reading a newspaper. graphic illustration, comic art, graphic novel art, vibrant, highly detailed",
    "a blue sks plush running in the jungle. graphic illustration, comic art, graphic novel art, vibrant, highly detailed",
    "a blue sks plush standing in a cave filled with gold. graphic illustration, comic art, graphic novel art, vibrant, highly detailed",
    "a blue sks plush driving a car filled with gold. graphic illustration, comic art, graphic novel art, vibrant, highly detailed",
    "a blue sks plush lying on a pile of gold. graphic illustration, comic art, graphic novel art, vibrant, highly detailed",
    "a blue sks plush standing on a beach at sunset. graphic illustration, comic art, graphic novel art, vibrant, highly detailed"
]

# StoryDiffusion에서 사용한거 그대로 가져옴. 자유롭게 변경 가능.
# e.g. 입이 계속 나온다면 "mouth"를 추가하는 등.
negative_prompt="photograph, deformed, glitch, noisy, realistic, stock photo, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",

# 변경 가능.
# 괜찮은 체크포인트 리스트: 900, 1250, 1400, 1450
lora_path = Path("/root/sangoh/CS492-Visual_Content_Generation/dreambooth/results/dreambooth-sdxl/nupjuki-new_data-color-lognormal/checkpoint-900")


timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
output_dir = os.path.join("./inference_outputs", lora_path.name, timestamp)
os.makedirs(output_dir, exist_ok=True)

config = {
    "timestamp": timestamp,
    "base_model_id": base_model_id,
    "num_inference_steps": num_inference_steps,
    "seed_start": seed,
    "prompts": prompts,
    "negative_prompt": negative_prompt,
    "lora_path": str(lora_path),
    "output_dir": output_dir
}

config_path = Path(output_dir) / "config.yaml"
with open(config_path, "w") as f:
    yaml.dump(config, f, allow_unicode=True)

print(f"[INFO] Saved configuration YAML: {config_path}")

# 경로를 문자열 리스트로 변환
lora_model_ids = [str(lora_path)]

# ===== 파이프라인 초기화 =====
print(f"[INFO] Loading base model: {base_model_id}")
pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

# ===== 생성 루프 =====
lora_name = Path(lora_path).name
print(f"\n[INFO] Loading LoRA: {lora_name}")

# LoRA 로드
pipe.load_lora_weights(lora_path, use_safetensors=True)

# 각 프롬프트에 대해 생성
for prompt_idx, prompt in enumerate(prompts):
    print(f"  - Generating with prompt {prompt_idx + 1}/{len(prompts)}: {prompt[:50]}...")
    
    # 시드 설정
    generator = torch.Generator(device="cuda").manual_seed(seed + prompt_idx)
    
    # 이미지 생성
    image = pipe(
        prompt=prompt,
        negative_prompt="photograph, deformed, glitch, noisy, realistic, stock photo, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
        num_inference_steps=num_inference_steps,
        generator=generator
    ).images[0]
    
    # 저장 (lora_name_promptX.png 형식)
    output_filename = f"{lora_name}_prompt{prompt_idx + 1:02d}.png"
    output_path = Path(output_dir) / output_filename
    image.save(output_path)
    print(f"    ✓ Saved: {output_path}")