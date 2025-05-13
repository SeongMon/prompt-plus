import argparse
from PIL import Image
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from diffusers.utils import is_xformers_available
from prompt_plus import TextualInversionStableDiffusionPipeline, PPlusStableDiffusionPipeline
import os
from utils.args_config_inference import parse_args
from utils.templates import imagenet_prompt_list, scene_prompts, costume_prompts
from torch import autocast

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def run_inference(prompts, mode, pipe, args):
    with torch.no_grad(), autocast(device_type=args.device if args.float16 else "cpu"):
        generator = None
        if args.seed:
            print(f"Using seed: {args.seed}")
            generator = torch.Generator(device=args.device).manual_seed(args.seed)
        for prompt in prompts:
            images = pipe(
                prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                num_images_per_prompt=args.num_images_per_prompt,
                height=args.height,
                width=args.width
            ).images
            
            grid_image = image_grid(images, 1, args.num_images_per_prompt)
            if not os.path.exists(f"{args.output_dir}/{mode}"):
                os.makedirs(f"{args.output_dir}/{mode}")
            grid_image.save(f"{args.output_dir}/{mode}/{prompt}.png")

def main():
    args = parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {args.device}")

    # task : p+, style_mixing
    if args.task == "p+":
        pipe = PPlusStableDiffusionPipeline.from_learned_embed(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            learned_embed_name_or_path=args.learned_embed_name_or_path,
            torch_dtype=torch.float16 if args.float16 else None,
        ).to(args.device)
    elif args.task == "style_mixing":
        pipe = PPlusStableDiffusionPipeline.from_learned_embed(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            learned_embed_name_or_path=[args.learned_embed_name_or_path, args.learned_embed_name_or_path_2],
            torch_dtype=torch.float16 if args.float16 else None,
            style_mixing_k_K=(args.mixing_start, args.mixing_end),
        ).to(args.device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    if is_xformers_available():
        pipe.enable_xformers_memory_efficient_attention()
       
    if args.prompt is not None:
        prompts = [args.prompt]
        run_inference(prompts, mode="user_prompts", pipe=pipe, args=args)
    elif args.prompts == "scene":
        prompts = scene_prompts(unique_token=args.placeholder_token)
        run_inference(prompts, mode="scene_prompts", pipe=pipe, args=args)
    elif args.prompts == "costume":
        prompts = costume_prompts(unique_token=args.placeholder_token)
        run_inference(prompts, mode="costume_prompts", pipe=pipe, args=args)
    elif args.prompts == "train":
        prompts = imagenet_prompt_list(args.placeholder_token)
        run_inference(prompts, mode="train", pipe=pipe, args=args)

if __name__ == '__main__':
    main()