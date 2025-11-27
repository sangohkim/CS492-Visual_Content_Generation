# Comic Generation with StoryDiffusion

이 가이드는 `Comic_Generation.py` 스크립트를 사용하여 일관된 캐릭터로 코믹을 생성하는 방법을 설명합니다.

## 빠른 시작

### 1. 기본 사용법

```bash
python Comic_Generation.py \
  --general_prompt "a man with a black suit" \
  --prompts "wake up in the bed" "have breakfast" "go to work" "work in office" \
  --output_dir ./my_comic
```

### 2. 사전 정의된 예제 실행

```bash
# 실행 권한 부여
chmod +x run_comic_generation.sh

# 예제 실행
./run_comic_generation.sh
```

### 3. 커스텀 코믹 생성

```bash
# 실행 권한 부여
chmod +x run_custom_comic.sh

# 스크립트 편집 (원하는 에디터 사용)
nano run_custom_comic.sh

# 실행
./run_custom_comic.sh
```

## 주요 파라미터 설명

### 필수 파라미터

- `--general_prompt`: 캐릭터의 기본 설명 (예: "a man with a black suit")
- `--prompts`: 각 패널의 장면 설명 (공백으로 구분)

### 스타일 설정

- `--style`: 이미지 스타일 (기본값: "Comic book")
  - 사용 가능한 스타일: Comic book, Anime, Photographic, Cinematic, 3D Model 등
  - 전체 목록: `utils/style_template.py` 참고

### Consistent Self-Attention 파라미터

- `--id_length`: 참조 이미지 개수 (기본값: 4)
  - 캐릭터 일관성 유지를 위한 기준 이미지 수
  - 권장: 3-4개

- `--sa32`: 32×32 해상도의 attention 강도 (기본값: 0.5)
  - 범위: 0.0 ~ 1.0
  - 높을수록 강한 일관성, 낮을수록 다양성

- `--sa64`: 16×16 해상도의 attention 강도 (기본값: 0.5)
  - 범위: 0.0 ~ 1.0
  - 높을수록 강한 일관성, 낮을수록 다양성

### 이미지 생성 파라미터

- `--num_steps`: 추론 단계 수 (기본값: 50)
  - 많을수록 품질 향상, 속도 감소
  - 권장: 30-50

- `--guidance_scale`: CFG 스케일 (기본값: 5.0)
  - 높을수록 프롬프트를 강하게 따름
  - 권장: 5.0-7.5

- `--height`, `--width`: 이미지 크기 (기본값: 1024×1024)
  - 권장: 768×768 또는 1024×1024

- `--seed`: 랜덤 시드 (기본값: 42)
  - 재현 가능한 결과를 위해 설정

### 모델 선택

- `--model`: 사용할 모델 (기본값: SDXL)
  - SDXL: 기본 모델
  - RealVision: 사실적인 이미지
  - Juggernaut: 고품질 이미지
  - Unstable: 실험적 모델

### VRAM 최적화

- `--use_sequential_offload`: Sequential CPU Offload 활성화
  - 최대 VRAM 절약 (가장 느림)
  - 권장: VRAM 부족 시

- `--use_attention_slicing`: Attention Slicing 활성화 (기본: 활성화)
  - 메모리 사용량 감소

- `--use_vae_slicing`: VAE Slicing 활성화
  - VAE 디코딩 메모리 감소

### 출력 설정

- `--output_dir`: 출력 디렉토리 (기본값: ./comic_outputs)
- `--font_path`: 폰트 파일 경로
- `--font_size`: 폰트 크기 (기본값: 30)

## 사용 예시

### 예시 1: 간단한 4패널 코믹

```bash
python Comic_Generation.py \
  --general_prompt "a cute cat" \
  --prompts "sleeping on a bed" "playing with yarn" "eating fish" "looking at the moon" \
  --output_dir ./outputs/cat_comic
```

### 예시 2: 고품질 설정

```bash
python Comic_Generation.py \
  --general_prompt "a superhero in a red cape" \
  --prompts "flying in the sky" "saving people" "fighting villains" "resting at home" \
  --style "Cinematic" \
  --id_length 3 \
  --sa32 0.7 \
  --sa64 0.7 \
  --num_steps 50 \
  --guidance_scale 7.0 \
  --height 1024 \
  --width 1024 \
  --output_dir ./outputs/superhero_comic
```

### 예시 3: VRAM 최적화 (낮은 메모리)

```bash
python Comic_Generation.py \
  --general_prompt "a young wizard" \
  --prompts "casting spell" "reading book" "brewing potion" "flying on broom" \
  --height 768 \
  --width 768 \
  --use_sequential_offload \
  --use_attention_slicing \
  --use_vae_slicing \
  --output_dir ./outputs/wizard_comic
```

### 예시 4: 많은 패널 (8개)

```bash
python Comic_Generation.py \
  --general_prompt "a robot explorer" \
  --prompts \
    "waking up in spaceship" \
    "checking equipment" \
    "landing on planet" \
    "exploring terrain" \
    "finding artifact" \
    "meeting alien" \
    "escaping danger" \
    "returning to ship" \
  --id_length 4 \
  --output_dir ./outputs/robot_adventure
```

## 출력 파일

생성된 결과물은 지정한 `output_dir`에 저장됩니다:

- `panel_00.png`, `panel_01.png`, ... : 개별 패널 이미지
- `comic_page_00.png`, `comic_page_01.png`, ... : 4패널 레이아웃으로 구성된 코믹 페이지

## 팁 & 트릭

### 캐릭터 일관성 향상

1. `id_length`를 3-4로 설정
2. `sa32`와 `sa64`를 0.6-0.7로 높이기
3. `general_prompt`에 상세한 캐릭터 설명 추가

### 메모리 부족 해결

1. `--use_sequential_offload` 활성화
2. `--use_attention_slicing` 활성화
3. 해상도를 768×768로 낮추기
4. `--use_vae_slicing` 활성화

### 생성 속도 향상

1. `--num_steps`를 30-40으로 낮추기
2. Sequential Offload 대신 Model Offload 사용 (플래그 제거)
3. 해상도 낮추기

### 품질 향상

1. `--num_steps`를 50-75로 늘리기
2. `--guidance_scale`을 7.0-7.5로 조정
3. 높은 해상도 사용 (1024×1024)
4. 상세한 프롬프트 작성

## 문제 해결

### Out of Memory (OOM) 에러
```bash
# 모든 최적화 활성화
--use_sequential_offload --use_attention_slicing --use_vae_slicing --height 768 --width 768
```

### 캐릭터가 일관되지 않음
```bash
# SA 강도 증가
--sa32 0.8 --sa64 0.8 --id_length 4
```

### 생성 속도가 너무 느림
```bash
# Sequential Offload 제거 및 Step 감소
--num_steps 30
```

### 이미지 품질이 낮음
```bash
# Step 증가 및 Guidance 조정
--num_steps 75 --guidance_scale 7.5
```

## 추가 정보

- 스타일 목록: `utils/style_template.py` 파일 참고
- 모델 정보: Hugging Face에서 각 모델 페이지 확인
- 논문: StoryDiffusion - Consistent Self-Attention for Long-Range Image and Video Generation
