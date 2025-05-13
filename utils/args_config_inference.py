import argparse
import os
from typing import Optional

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, help="model name or path", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--learned_embed_name_or_path", type=str, help="model path for learned embedding")
    parser.add_argument("--learned_embed_name_or_path_2", type=str, help="model path for learned embedding")
    parser.add_argument("--placeholder_token", type=str, help="placeholder token", required=True)
    parser.add_argument("--task", type =str, help="task name", choices=["p+", "style_mixing"], default="p+")
    parser.add_argument("--original_pipe", action="store_true", help="load standard pipeline")
    parser.add_argument("--device", type=str, help="Device on which Stable Diffusion will be run", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--float16", action="store_true", help="load float16")
    # diffusers config
    parser.add_argument("--num_inference_steps", type=int, default=50, help="number of ddim sampling steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="unconditional guidance scale")
    parser.add_argument("--num_images_per_prompt", type=int, default=4, help="number of images per prompt")
    parser.add_argument("--height", type=int, default=512, help="image height, in pixel space",)
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space",)
    parser.add_argument("--seed", type=int, default=None, help="the seed (for reproducible sampling)")
    parser.add_argument("--output_dir", type=str, default=".", help="the output directory where the model predictions and checkpoints will be written")
    parser.add_argument("--mixing_start", type=int, default=0, help="the start layer for style mixing")
    parser.add_argument("--mixing_end", type=int, default=0, help="the end layer for style mixing")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--prompt", type=str, default=None, help="the prompt to render")
    group.add_argument("--prompts", type=str, choices=["scene", "costume", "train"], default="scene", help="choose prompt set: scene, costume, train")
    opt = parser.parse_args()
    return opt