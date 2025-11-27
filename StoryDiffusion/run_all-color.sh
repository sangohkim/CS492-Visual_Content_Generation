#!/bin/bash

# parent directory path (여기만 바꿔서 사용하면 됨)
parent_dir="/root/sangoh/CS492-Visual_Content_Generation/dreambooth/results/dreambooth-sdxl/nupjuki-new_data-color-lognormal"

# Minimum checkpoint number to run (optional first argument). Defaults to 0.
MIN_CKPT=750
echo "Minimum checkpoint number to run: $MIN_CKPT"

# checkpoint-* 패턴과 일치하는 디렉터리들을 모두 탐색
for ckpt in "$parent_dir"/checkpoint-*; do
    # 디렉터리가 맞는 경우에만 실행
    if [ -d "$ckpt" ]; then
        base=$(basename "$ckpt")
        # extract number after 'checkpoint-'
        num=${base#checkpoint-}

        # if not a number, skip
        if ! [[ "$num" =~ ^[0-9]+$ ]]; then
            echo "Skipping non-numeric checkpoint: $base"
            continue
        fi

        # if checkpoint number is NOT greater than MIN_CKPT, skip
        if [ "$num" -le "$MIN_CKPT" ]; then
            echo "Skipping $base (<= $MIN_CKPT)"
            continue
        fi

        echo "Running run.sh with checkpoint: $ckpt"
        bash run_comic_generation_dreambooth.sh "$ckpt"
    fi
done
