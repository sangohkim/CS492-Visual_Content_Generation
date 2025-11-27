#!/bin/bash

# parent directory path (여기만 바꿔서 사용하면 됨)
parent_dir="/root/sangoh/CS492-Visual_Content_Generation/dreambooth/results/dreambooth-sdxl/nupjuki-new_data"

# checkpoint-* 패턴과 일치하는 디렉터리들을 모두 탐색
for ckpt in "$parent_dir"/checkpoint-*; do
    # 디렉터리가 맞는 경우에만 실행
    if [ -d "$ckpt" ]; then
        echo "Running run.sh with checkpoint: $ckpt"
        bash run_comic_generation_dreambooth-no_color.sh "$ckpt"
    fi
done
