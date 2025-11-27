#!/usr/bin/env python
# coding=utf-8
"""
Comic Generation Script using StoryDiffusion
This script generates consistent comic images using StoryDiffusion's Consistent Self-Attention mechanism.
"""

import argparse
import numpy as np
import torch
import random
import os
import copy
from PIL import Image, ImageFont
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
import torch.nn.functional as F
from utils.gradio_utils import is_torch2_available, cal_attn_mask_xl
from utils.utils import get_comic_4panel
from utils.style_template import styles

if is_torch2_available():
    from utils.gradio_utils import AttnProcessor2_0 as AttnProcessor
else:
    from utils.gradio_utils import AttnProcessor


# Global variables for consistent self-attention mechanism
# Attention step tracking:
global attn_count        # Current attention processor call count within a diffusion step
global total_count       # Total number of attention processors that use consistent self-attention
global cur_step          # Current diffusion step (0 to num_steps)

# Image batch configuration:
global id_length         # Number of identity/reference images to generate and store features from
global total_length      # Total number of images in current batch (id_length + 1 for batched generation)

# Generation phase control:
global write             # True: storing identity features (write phase), False: using stored features (read phase)

# Consistent self-attention strength parameters:
global sa32              # Self-attention strength for 32x32 resolution (range: 0-1)
global sa64              # Self-attention strength for 64x64 resolution (range: 0-1)

# Image dimensions:
global height            # Height of generated images in pixels
global width             # Width of generated images in pixels

# Model components:
global attn_procs        # Dictionary of attention processors for UNet
global unet              # The UNet model from diffusion pipeline
global mask1024          # Attention mask for 32x32 resolution (1024 = 32*32)
global mask4096          # Attention mask for 64x64 resolution (4096 = 64*64)

# Legacy (unused):
global cur_model_type    # Model type identifier (currently unused)


class SpatialAttnProcessor2_0(torch.nn.Module):
    """Attention processor for Consistent Self-Attention"""
    
    def __init__(self, hidden_size=None, cross_attention_dim=None, id_length=4, device="cuda", dtype=torch.float16):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.total_length = id_length + 1
        self.id_length = id_length
        self.id_bank = {}

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        global total_count, attn_count, cur_step, mask1024, mask4096
        global sa32, sa64, write, height, width
        
        if write:
            self.id_bank[cur_step] = [hidden_states[:self.id_length], hidden_states[self.id_length:]]
        else:
            encoder_hidden_states = torch.cat((
                self.id_bank[cur_step][0].to(self.device),
                hidden_states[:1],
                self.id_bank[cur_step][1].to(self.device),
                hidden_states[1:]
            ))
        
        if cur_step < 5:
            # Early diffusion steps: use standard self-attention to build basic structure
            hidden_states = self.__call2__(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        else:
            random_number = random.random()
            rand_num = 0.3 if cur_step < 20 else 0.1
            
            if random_number > rand_num:
                # Consistent self-attention across images with probability 1 - rand_num
                if not write:
                    # New images attend to the ID images
                    if hidden_states.shape[1] == (height//32) * (width//32):
                        attention_mask = mask1024[mask1024.shape[0] // self.total_length * self.id_length:]
                    else:
                        attention_mask = mask4096[mask4096.shape[0] // self.total_length * self.id_length:]
                else:
                    if hidden_states.shape[1] == (height//32) * (width//32):
                        attention_mask = mask1024[:mask1024.shape[0] // self.total_length * self.id_length, :mask1024.shape[0] // self.total_length * self.id_length]
                    else:
                        attention_mask = mask4096[:mask4096.shape[0] // self.total_length * self.id_length, :mask4096.shape[0] // self.total_length * self.id_length]
                hidden_states = self.__call1__(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
            else:
                # Standard self-attention to maintain diversity
                hidden_states = self.__call2__(attn, hidden_states, None, attention_mask, temb)
        
        attn_count += 1
        if attn_count == total_count:
            attn_count = 0
            cur_step += 1
            mask1024, mask4096 = cal_attn_mask_xl(
                self.total_length, self.id_length, sa32, sa64, height, width,
                device=self.device, dtype=self.dtype
            )
        
        return hidden_states

    def __call1__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            total_batch_size, channel, h, w = hidden_states.shape
            hidden_states = hidden_states.view(total_batch_size, channel, h * w).transpose(1, 2)
        
        total_batch_size, nums_token, channel = hidden_states.shape
        img_nums = total_batch_size // 2
        hidden_states = hidden_states.view(-1, img_nums, nums_token, channel).reshape(-1, img_nums * nums_token, channel)
        batch_size, sequence_length, _ = hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            encoder_hidden_states = encoder_hidden_states.view(-1, self.id_length+1, nums_token, channel).reshape(-1, (self.id_length+1) * nums_token, channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(total_batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(total_batch_size, channel, h, w)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states

    def __call2__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        """
        Standard self-attention without cross-image consistency.
        
        This method is used in two scenarios:
        1. Early diffusion steps (cur_step < 5): To establish basic structure without enforcing consistency
        2. Randomly during later steps (with small probability): To maintain generation diversity
        
        Unlike __call1__ which performs consistent attention across multiple images,
        this method processes each image independently, similar to vanilla SDXL attention.
        
        Args:
            attn: Attention module from UNet
            hidden_states: Current feature maps [batch, tokens, channels] or [batch, channels, h, w]
            encoder_hidden_states: Optional identity features from ID bank (usually None for standard attention)
            attention_mask: Optional attention mask
            temb: Time embedding (unused in this implementation)
        
        Returns:
            hidden_states: Processed features with same shape as input
        """
        # Store residual for skip connection
        residual = hidden_states

        # Step 1: Apply spatial normalization if available (for spatial transformer blocks)
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        # Step 2: Reshape spatial features to sequence format if needed
        # Convert [batch, channels, h, w] → [batch, h*w, channels]
        if input_ndim == 4:
            batch_size, channel, h, w = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, h * w).transpose(1, 2)

        batch_size, sequence_length, channel = hidden_states.shape
        
        # Step 3: Prepare attention mask for multi-head attention
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        # Step 4: Apply group normalization if available
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # Step 5: Compute Query from current features
        query = attn.to_q(hidden_states)

        # Step 6: Compute Key and Value
        # If encoder_hidden_states is provided (from ID bank), use it for K/V
        # Otherwise, use self-attention (Q, K, V all from hidden_states)
        if encoder_hidden_states is None:
            # Pure self-attention: each image attends to itself only
            encoder_hidden_states = hidden_states
        else:
            # Cross-attention with ID bank features (rarely used in __call2__)
            # Reshape to include all identity images in the sequence
            encoder_hidden_states = encoder_hidden_states.view(
                -1, self.id_length + 1, sequence_length, channel
            ).reshape(-1, (self.id_length + 1) * sequence_length, channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Step 7: Reshape for multi-head attention
        # Split channels into multiple heads: [batch, seq, channels] → [batch, heads, seq, head_dim]
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Step 8: Compute scaled dot-product attention
        # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # Step 9: Reshape back to original format
        # [batch, heads, seq, head_dim] → [batch, seq, channels]
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Step 10: Apply output projection (linear layer + dropout)
        hidden_states = attn.to_out[0](hidden_states)  # Linear projection
        hidden_states = attn.to_out[1](hidden_states)  # Dropout

        # Step 11: Reshape back to spatial format if input was spatial
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, h, w)

        # Step 12: Apply residual connection (skip connection)
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        # Step 13: Rescale output (for training stability)
        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def setup_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def apply_style(style_name, positives, negative=""):
    """Apply style template to prompts"""
    p, n = styles.get(style_name, styles["(No style)"])
    return [p.replace("{prompt}", positive) for positive in positives], n + ' ' + negative


def apply_style_positive(style_name, positive):
    """Apply style template to a single prompt"""
    p, n = styles.get(style_name, styles["(No style)"])
    return p.replace("{prompt}", positive)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate comics using StoryDiffusion")
    
    # Model settings
    parser.add_argument("--model", type=str, default="SDXL",
                       choices=["SDXL", "RealVision", "Juggernaut", "Unstable"],
                       help="Model to use for generation")
    
    # Generation settings
    parser.add_argument("--general_prompt", type=str, required=True,
                       help="General character description (e.g., 'a man with a black suit')")
    parser.add_argument("--prompts", type=str, nargs='+', required=True,
                       help="Scene descriptions for each panel")
    parser.add_argument("--style", type=str, default="Comic book",
                       help="Style to apply (see utils/style_template.py)")
    parser.add_argument("--negative_prompt", type=str,
                       default="naked, deformed, bad anatomy, disfigured, poorly drawn face, mutation, extra limb, ugly, disgusting, poorly drawn hands, missing limb, floating limbs, disconnected limbs, blurry, watermarks, oversaturated, distorted hands, amputation",
                       help="Negative prompt")
    
    # Consistent self-attention parameters
    parser.add_argument("--id_length", type=int, default=4,
                       help="Number of identity images to generate")
    parser.add_argument("--sa32", type=float, default=0.5,
                       help="Strength of consistent self-attention for 32x32 resolution (0-1)")
    parser.add_argument("--sa64", type=float, default=0.5,
                       help="Strength of consistent self-attention for 16x16 resolution (0-1)")
    
    # Image generation parameters
    parser.add_argument("--num_steps", type=int, default=50,
                       help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=5.0,
                       help="Guidance scale for classifier-free guidance")
    parser.add_argument("--height", type=int, default=1024,
                       help="Height of generated images")
    parser.add_argument("--width", type=int, default=1024,
                       help="Width of generated images")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    # VRAM optimization
    parser.add_argument("--use_sequential_offload", action="store_true",
                       help="Use sequential CPU offload (slower but uses least VRAM)")
    parser.add_argument("--use_attention_slicing", action="store_true", default=True,
                       help="Enable attention slicing")
    parser.add_argument("--use_vae_slicing", action="store_true",
                       help="Enable VAE slicing")
    
    # Output settings
    parser.add_argument("--output_dir", type=str, default="./comic_outputs",
                       help="Directory to save generated comics")
    parser.add_argument("--font_path", type=str,
                       default="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                       help="Path to font file for comic text")
    parser.add_argument("--font_size", type=int, default=30,
                       help="Font size for comic captions")
    
    return parser.parse_args()


def main():
    global attn_count, total_count, id_length, total_length, cur_step, cur_model_type
    global write, sa32, sa64, height, width
    global attn_procs, unet, mask1024, mask4096
    
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments to file
    import json
    args_dict = vars(args)
    args_file = os.path.join(args.output_dir, "generation_args.json")
    with open(args_file, 'w') as f:
        json.dump(args_dict, f, indent=4)
    print(f"Arguments saved to {args_file}")
    
    # Initialize global variables
    attn_count = 0
    total_count = 0
    cur_step = 0
    id_length = args.id_length
    total_length = len(args.prompts) + 1
    cur_model_type = ""
    device = "cuda"
    attn_procs = {}
    write = False
    sa32 = args.sa32
    sa64 = args.sa64
    height = args.height
    width = args.width
    
    print("=" * 80)
    print("Comic Generation with StoryDiffusion")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"General Prompt: {args.general_prompt}")
    print(f"Number of Panels: {len(args.prompts)}")
    print(f"Style: {args.style}")
    print(f"Image Size: {args.width}x{args.height}")
    print(f"ID Length: {args.id_length}")
    print(f"SA32/SA64: {args.sa32}/{args.sa64}")
    print(f"Seed: {args.seed}")
    print("=" * 80)
    
    # Setup seed
    setup_seed(args.seed)
    
    # Load model
    models_dict = {
        "Juggernaut": "RunDiffusion/Juggernaut-XL-v8",
        "RealVision": "SG161222/RealVisXL_V4.0",
        "SDXL": "stabilityai/stable-diffusion-xl-base-1.0",
        "Unstable": "stablediffusionapi/sdxl-unstable-diffusers-y"
    }
    assert args.model == "SDXL", "Only SDXL model is supported"
    
    print("\n[1/5] Loading Stable Diffusion XL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        models_dict[args.model],
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    
    pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(args.num_steps)
    unet = pipe.unet
    
    # Insert Consistent Self-Attention
    print("[2/5] Setting up Consistent Self-Attention...")
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if cross_attention_dim is None and name.startswith("up_blocks"):
            attn_procs[name] = SpatialAttnProcessor2_0(id_length=id_length)
            total_count += 1
        else:
            attn_procs[name] = AttnProcessor()
    
    print(f"✓ Loaded consistent self-attention ({total_count} processors)")
    unet.set_attn_processor(copy.deepcopy(attn_procs))
    
    # Apply VRAM optimizations
    print("[3/5] Applying VRAM optimizations...")
    if args.use_sequential_offload:
        pipe.enable_sequential_cpu_offload()
        print("✓ Sequential CPU Offload enabled")
    else:
        pipe.enable_model_cpu_offload()
        print("✓ Model CPU Offload enabled")
    
    if args.use_attention_slicing:
        pipe.enable_attention_slicing(1)
        print("✓ Attention Slicing enabled")
    
    if args.use_vae_slicing:
        pipe.enable_vae_slicing()
        print("✓ VAE Slicing enabled")
    
    # Initialize attention masks
    mask1024, mask4096 = cal_attn_mask_xl(
        total_length, id_length, sa32, sa64, height, width,
        device=device, dtype=torch.float16
    )  # [total_length=len(prompts)+1 * (height // 32) * (width // 32), total_length * (height // 32) * (width // 32)]
    
    # Prepare prompts
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    full_prompts = [f"{args.general_prompt}, {prompt}" for prompt in args.prompts]
    id_prompts = full_prompts[:id_length]
    real_prompts = full_prompts[id_length:]
    
    # Generate identity images
    print(f"\n[4/5] Generating {id_length} identity images...")
    torch.cuda.empty_cache()
    write = True
    cur_step = 0
    attn_count = 0
    
    id_prompts_styled, negative_prompt_styled = apply_style(args.style, id_prompts, args.negative_prompt)
    
    id_images = pipe(
        id_prompts_styled,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        height=height,
        width=width,
        negative_prompt=negative_prompt_styled,
        generator=generator
    ).images
    
    print(f"✓ Generated {len(id_images)} identity images")
    
    # Generate remaining images
    write = False
    real_images = []
    
    if len(real_prompts) > 0:
        print(f"[5/5] Generating {len(real_prompts)} additional panel images...")
        for idx, real_prompt in enumerate(real_prompts):
            cur_step = 0
            real_prompt_styled = apply_style_positive(args.style, real_prompt)
            print(f"  Generating panel {id_length + idx + 1}/{len(full_prompts)}...")
            
            image = pipe(
                real_prompt_styled,
                num_inference_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                height=height,
                width=width,
                negative_prompt=negative_prompt_styled,
                generator=generator
            ).images[0]
            
            real_images.append(image)
            torch.cuda.empty_cache()
    
    print(f"✓ Generated {len(real_images)} additional images")
    
    # Combine all images
    total_images = id_images + real_images
    
    # Save individual images
    print(f"\nSaving individual images to {args.output_dir}...")
    for idx, img in enumerate(total_images):
        img.save(os.path.join(args.output_dir, f"panel_{idx:02d}.png"))
    print(f"✓ Saved {len(total_images)} individual images")
    
    # Create comic layout
    print("\nCreating comic layout...")
    try:
        font = ImageFont.truetype(args.font_path, args.font_size)
    except:
        print(f"⚠ Could not load font from {args.font_path}, using default")
        font = ImageFont.load_default()
    
    comics = get_comic_4panel(total_images, captions=args.prompts, font=font, pad_image=Image.open("/root/sangoh/CS492-Visual_Content_Generation/StoryDiffusion/images/pad_images.png"))
    
    # Save comics
    for idx, comic in enumerate(comics):
        comic_path = os.path.join(args.output_dir, f"comic_page_{idx:02d}.png")
        comic.save(comic_path)
        print(f"✓ Saved comic page: {comic_path}")
    
    print("\n" + "=" * 80)
    print("Comic generation completed successfully!")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
