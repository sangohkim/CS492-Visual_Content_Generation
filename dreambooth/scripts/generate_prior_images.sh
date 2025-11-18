#!/bin/bash

python generate_prior_images.py \
    --output_dir ./class_images/toy \
    --class_prompt "A toy" \
    --num_images 1000 \
    --torch_dtype bf16