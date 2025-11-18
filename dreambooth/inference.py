from huggingface_hub.repocard import RepoCard
from diffusers import DiffusionPipeline
import torch
from pathlib import Path
import re

# ===== 설정 =====
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
output_dir = "./inference_outputs"

# 체크포인트 디렉토리에서 자동으로 모든 체크포인트 찾기
checkpoint_base_dir = Path("/root/sangoh/CS492-Visual_Content_Generation/results/dreambooth-sdxl/monster_toy")

# checkpoint-N 형식의 모든 디렉토리 찾기
checkpoint_dirs = sorted(
    [d for d in checkpoint_base_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
    key=lambda x: int(re.search(r'checkpoint-(\d+)', x.name).group(1))
)

# 경로를 문자열 리스트로 변환
lora_model_ids = [str(d) for d in checkpoint_dirs]

print(f"[INFO] Found {len(lora_model_ids)} checkpoints:")
for i, path in enumerate(lora_model_ids, 1):
    print(f"  {i}. {Path(path).name}")

# 프롬프트 리스트
prompts = [
    "A sks monster running in the jungle",
    "A sks monster wearing a hat",
    "A sks monster jumping on the moon",
    "A sks monster playing with a ball in the park",
    "A sks monster standing on a beach at sunset",
]

# 추론 설정
num_inference_steps = 25
seed = 42

# ===== 파이프라인 초기화 =====
print(f"[INFO] Loading base model: {base_model_id}")
pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

# VAE를 float32로 설정 (안정성)
# pipe.vae.to(torch.float32)

# 출력 디렉토리 생성
Path(output_dir).mkdir(parents=True, exist_ok=True)

# ===== 생성 루프 =====
for lora_idx, lora_path in enumerate(lora_model_ids):
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
            num_inference_steps=num_inference_steps,
            generator=generator
        ).images[0]
        
        # 저장 (lora_name_promptX.png 형식)
        output_filename = f"{lora_name}_prompt{prompt_idx + 1:02d}.png"
        output_path = Path(output_dir) / output_filename
        image.save(output_path)
        print(f"    ✓ Saved: {output_path}")
    
    # LoRA 언로드 (다음 LoRA를 위해)
    pipe.unload_lora_weights()

print(f"\n[INFO] All done! Images saved to: {output_dir}")