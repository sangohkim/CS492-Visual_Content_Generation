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

prompts = []

# cand1 = [
#     "a blue sks plush driving a tiny red sports car.",
#     "a blue sks plush driving a futuristic hover vehicle.",
#     "a blue sks plush driving a classic vintage truck.",
#     "a blue sks plush driving a neon-lit cyberpunk car.",
#     "a blue sks plush driving a rugged off-road jeep.",
#     "a blue sks plush driving a small wooden cart through a forest.",
#     "a blue sks plush driving a submarine under glowing waters.",
#     "a blue sks plush driving a rocket-powered scooter.",
#     "a blue sks plush driving a golden chariot pulled by horses.",
#     "a blue sks plush driving a giant mechanical beetle.",
#     "a blue sks plush driving a spaceship through the stars.",
#     "a blue sks plush driving a steam-powered vehicle.",
#     "a blue sks plush driving a miniature train on a toy track.",
#     "a blue sks plush driving a futuristic monorail.",
#     "a blue sks plush driving a jet-powered race car.",
#     "a blue sks plush driving a huge armored tank.",
#     "a blue sks plush driving a magical pumpkin carriage.",
#     "a blue sks plush driving a snowmobile across icy terrain.",
#     "a blue sks plush driving a glowing crystal vehicle.",
#     "a blue sks plush driving a drone-like flying pod.",
#     "a blue sks plush driving a neon motorcycle through rain.",
#     "a blue sks plush driving a safari jeep across the savannah.",
#     "a blue sks plush driving a robotic insect-like machine.",
#     "a blue sks plush driving a compact retro mini car.",
#     "a blue sks plush driving a floating spherical vehicle.",
#     "a blue sks plush driving a sleek futuristic taxi.",
#     "a blue sks plush driving a dune buggy across the desert.",
#     "a blue sks plush driving a tiny boat through a calm river.",
#     "a blue sks plush driving a high-tech armored vehicle.",
#     "a blue sks plush driving a colorful parade float.",
#     "a blue sks plush driving a rocket car on a runway.",
#     "a blue sks plush driving a magical sleigh across the sky.",
#     "a blue sks plush driving a futuristic race pod.",
#     "a blue sks plush driving a moon rover on lunar soil.",
#     "a blue sks plush driving a giant hamster ball car.",
#     "a blue sks plush driving a heavy construction bulldozer.",
#     "a blue sks plush driving a crane truck in a work zone.",
#     "a blue sks plush driving a tractor across a farm field.",
#     "a blue sks plush driving a colorful ice cream truck.",
#     "a blue sks plush driving a glowing hover bike.",
#     "a blue sks plush driving a vintage convertible by the ocean.",
#     "a blue sks plush driving a shielded military jeep.",
#     "a blue sks plush driving a floating lotus-shaped vehicle.",
#     "a blue sks plush driving a magical cloud chariot.",
#     "a blue sks plush driving a compact go-kart on a race track.",
#     "a blue sks plush driving a futuristic armored rover.",
#     "a blue sks plush driving a shimmering prism-shaped car.",
#     "a blue sks plush driving an underwater bubble car.",
#     "a blue sks plush driving a neon-trimmed hover van.",
#     "a blue sks plush driving a rocket-powered unicycle."
# ]
# prompts.extend(cand1)

cand2 = [
    "a surprised blue sks plush, hands on cheeks, standing on wooden floor, golden illumination."
]

prompts.extend(cand2)

positive_prompt = " graphic illustration, comic art, graphic novel art, vibrant, highly detailed"

# StoryDiffusion에서 사용한거 그대로 가져옴. 자유롭게 변경 가능.
# e.g. 입이 계속 나온다면 "mouth"를 추가하는 등.
negative_prompt="photograph, deformed, glitch, noisy, realistic, stock photo, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
# negative_prompt="photograph, deformed, glitch, noisy, realistic, stock photo, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, mouth, smile, lips, ears",

# 변경 가능.
# 괜찮은 체크포인트 리스트: 900, 1250, 1400, 1450
lora_path = Path("/root/sangoh/CS492-Visual_Content_Generation/dreambooth/results/dreambooth-sdxl/nupjuki-new_data-color-lognormal/checkpoint-1450")


timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
output_dir = os.path.join("./inference_finals", lora_path.name, timestamp)
os.makedirs(output_dir, exist_ok=True)

config = {
    "timestamp": timestamp,
    "base_model_id": base_model_id,
    "num_inference_steps": num_inference_steps,
    "seed_start": seed,
    "prompts": prompts,
    "negative_prompt": negative_prompt,
    "lora_path": str(lora_path),
    "output_dir": output_dir,
    "positive_prompt": positive_prompt
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
    prompt = prompt + positive_prompt
    print(f"  - Generating with prompt {prompt_idx + 1}/{len(prompts)}: {prompt[:50]}...")
    
    # 시드 설정
    # generator = torch.Generator(device="cuda").manual_seed(seed + prompt_idx)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    
    # 이미지 생성
    image = pipe(
        prompt=prompt,
        negative_prompt="photograph, deformed, glitch, noisy, realistic, stock photo, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
        num_inference_steps=num_inference_steps,
        generator=generator
    ).images[0]
    
    # 저장 (lora_name_promptX.png 형식)
    output_filename = f"{lora_name}_prompt{prompt_idx + 1:02d}.jpg"
    output_path = Path(output_dir) / output_filename
    image.save(output_path, format="JPEG")
    print(f"    ✓ Saved: {output_path}")