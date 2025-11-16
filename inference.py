from huggingface_hub.repocard import RepoCard
from diffusers import DiffusionPipeline
import torch

lora_model_id = "/root/sangoh/CS492-Visual_Content_Generation/results/dreambooth-sdxl/monster_toy/checkpoint-1000"
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"

pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

pipe.load_lora_weights(lora_model_id, use_safetensors=True)
image = pipe("A picture of a sks monster running in a jungle", num_inference_steps=25).images[0]
image.save("sks-monster_toy.png")