#!/bin/bash

cd "$(dirname "$0")"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_BASE="./outputs_sangoh/run_${TIMESTAMP}"

LORA_PATH=$1
SEED=42

echo "Using LoRA checkpoint path: $LORA_PATH"

if [ ! -d "$LORA_PATH" ]; then
    echo "Error: LoRA checkpoint directory not found: $LORA_PATH"
    exit 1
fi

###########################################
# 공통으로 모든 스토리 앞에 붙일 기본 프롬프트
###########################################
base_prompts=(
  "standing in a lush forest illuminated by fireflies"
  "standing in a sunlit field of tall grass"
)

###########################################
# 여러 스토리 정의
# 구분자는 |
###########################################
stories=(
"running in a jungle|standing in a sunlit field of tall grass|standing in a tranquil Japanese-style garden|standing on a beach at sunset"
"standing at the entrance of a glowing forest|walking along a soft path through floating particles of light|running past tall crystal formations|standing on a quiet hill under a bright magical sky"
"standing on a flat rocky moon surface under distant stars|walking toward a bright crater edge|running across a low-gravity ridge|standing near a small platform overlooking several planets"
"standing between tall neon buildings in a quiet future city|walking along a clean glass walkway above the streets|running across a wide metallic bridge|standing in a bright open plaza surrounded by geometric towers"
"standing beside a large ancient stone arch|walking through a field of shimmering grass|running up a narrow path toward a glowing cliffside|standing on a cliff overlooking a calm fantasy valley"
"standing on a silent asteroid plain|walking past scattered metallic terrain features|running across a smooth dust field with clear star light|standing near the horizon where a large planet rises"
"standing in a forest filled with floating lantern-like lights|walking through a wide clearing with gentle glowing mist|running across a stone bridge in a quiet magical canyon|standing near a calm lake reflecting a bright fantasy moon"
"standing in front of a glowing transit terminal|walking through a long corridor lined with light panels|running along a narrow elevated path|standing at the center of a quiet futuristic square"
"standing on a quiet space outpost walkway|walking along a long metallic path under a dark sky|running toward a tall communication tower structure|standing on a platform with a wide view of distant galaxies"
"standing in a large open field under a clear bright sky|walking toward a distant ridge|running along a smooth path with soft shadows|standing on a small hill overlooking the landscape"
)

###########################################
# 각 스토리를 순차적으로 실행
###########################################
story_idx=1
for story in "${stories[@]}"; do
    
    echo "=================================================="
    echo "Processing Story $story_idx ..."
    echo "=================================================="

    # 스토리 문자열을 배열로 변환
    IFS='|' read -r -a story_prompts <<< "$story"

    # base_prompts + story_prompts 결합
    combined_prompts=("${base_prompts[@]}" "${story_prompts[@]}")

    # 출력 디렉토리
    STORY_OUTPUT="${OUTPUT_BASE}/story_${story_idx}"
    mkdir -p "$STORY_OUTPUT"

    # ======== Comic Generation 실행 ========
    python Comic_Generation_dreambooth.py \
      --general_prompt "a blue sks plush" \
      --prompts "${combined_prompts[@]}" \
      --lora_path "$LORA_PATH" \
      --style "Comic book storydiffv2" \
      --id_length 2 \
      --sa32 0.5 \
      --sa64 0.5 \
      --num_steps 50 \
      --guidance_scale 5.0 \
      --height 1024 \
      --width 1024 \
      --output_dir "$STORY_OUTPUT" \
      --use_sequential_offload \
      --use_attention_slicing \
      --seed $SEED

    ((story_idx++))
done

echo "=================================================="
echo "All stories processed successfully!"
echo "Output directory: $OUTPUT_BASE"
echo "=================================================="
