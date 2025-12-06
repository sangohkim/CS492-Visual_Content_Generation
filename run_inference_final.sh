#!/bin/bash

ckpt_path=../dreambooth/results/dreambooth-sdxl/nupjuki-new_data-color-lognormal/checkpoint-1450

python dreambooth/inference_comics_single_final.py

bash StoryDiffusion/run_comic_generation_dreambooth_final.sh $ckpt_path

orig_prompts=(
    "a blue sks plush, reading a newspaper in a vintage cafe, colorful."
    "a blue sks plush, walking out of a cafe, colorful."
    "cars driving in a lush forest illuminated by fireflies, whole view",
    "a blue sks plush, standing in a lush forest illuminated by fireflies"
    "a giant spider in a lush forest illuminated by fireflies"
    "a blue sks plush running in a lush forest"
    "a house in a lush forest illuminated by fireflies"
    "a room, filled with gold, ancient artifacts, and treasure chests"
    "a blue sks plush surrounded by money, jewels, wearing sunglasses, wearing golden clothes."
)

final_prompts=(
    "reading a newspaper in a vintage cafe, colorful."
    "walking out of a cafe, colorful."
    "cars driving in a lush forest illuminated by fireflies, whole view",
    "standing in a lush forest illuminated by fireflies"
    "a giant spider in a lush forest illuminated by fireflies"
    "running in a lush forest"
    "a house in a lush forest illuminated by fireflies"
    "a room, filled with gold, ancient artifacts, and treasure chests"
    "surrounded by money, jewels, wearing sunglasses, wearing golden clothes."
)

img_paths=(
    "inference_outputs_final/checkpoint-1450_prompt01.jpg"
    "inference_outputs_final/checkpoint-1450_prompt02.jpg"
    "inference_outputs_final/panel_02.png"
    "inference_outputs_final/panel_00.png"
    "inference_outputs_final/panel_04.png"
    "inference_outputs_final/panel_03.png"
    "inference_outputs_final/panel_05.png"
    "inference_outputs_final/panel_06.png"
    "inference_outputs_final/checkpoint-1450_prompt03.jpg"
)

# python StoryDiffusion/generate_4panels.py \
#   --image_paths ${img_paths[@]} \
#   --prompts "${final_prompts[@]}" \
#   --output_dir "inference_outputs_final" \
#   --font_size 30

python StoryDiffusion/add_captions.py \
  --image_paths ${img_paths[@]} \
  --prompts "${final_prompts[@]}" \
  --output_dir "inference_outputs_final" \
  --font_size 30