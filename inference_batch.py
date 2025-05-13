import argparse
import os
import random
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import AutoTokenizer
import sys

from models.neti_like.neti_clip_text_encoder import NETICLIPTextModel, NETICLIPTextEmbeddings
from models.neti_like.neti_mixture_of_embedding import NETIMixtureOfEmbedding
from models.prompt_plus import PPlusUNet2DConditionModel
from utils import img_captions
from utils.templates import prompt_list_2, imagenet_prompt_list, prompt_list_1
from utils.types import MOEBatch
from utils.args_config_inference import parse_args

import argparse
from PIL import Image
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from diffusers.utils import is_xformers_available
from prompt_plus import TextualInversionStableDiffusionPipeline, PPlusStableDiffusionPipeline
import os
from utils.args_config_inference import parse_args


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def make_batch_prompts(prompts):
    batch_prompts = []
    tmp_prompts = []
    for idx, prompt in enumerate(prompts):
        # print(len(prompts))
        tmp_prompts.append(prompt)
        if len(tmp_prompts) == args.batch_size:
            batch_prompts.append(tmp_prompts)
            tmp_prompts = []
        elif idx == len(prompts) - 1:
            batch_prompts.append(tmp_prompts)
    return batch_prompts

def run_inference(prompts, mode, tokenizer_l2_norms_mean=None):
    seeds = [42, 43]
    for seed in seeds:
        batch_prompts_list = make_batch_prompts(prompts)
        for idx, batch_prompts in enumerate(batch_prompts_list):
            if idx == len(batch_prompts_list) - 1:
                batch_size = len(batch_prompts_list[idx])
            else:
                batch_size = args.batch_size
            print(batch_size)
            # Init latents
            set_seed(seed)  # 하나의 seed로 랜덤성 고정
            vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
            latents = torch.randn(
                (batch_size, unet.config.in_channels, 512 // vae_scale_factor, 512 // vae_scale_factor),
                device=device
            ) * scheduler.init_noise_sigma

            scheduler.set_timesteps(50, device=device)
            # breakpoint()
            # Text input batch
            text_inputs = tokenizer(batch_prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
            uncond_inputs = tokenizer([""] * batch_size, padding="max_length", max_length=77, return_tensors="pt")
            text_inputs = text_inputs.to(device)
            uncond_inputs = uncond_inputs.to(device)

            # Sampling loop
            for t in tqdm(scheduler.timesteps):
                if args.tokenizer_rescale:
                    tokenizer_l2_norms_mean = tokenizer_l2_norms_mean.to(device)
                    tokenizer_l2_norms_mean_batch = tokenizer_l2_norms_mean.repeat(batch_size)
                else:
                    tokenizer_l2_norms_mean_batch = None
                # breakpoint()
                if args.layer_wise:
                    # 16개 layer 전부 batch 처리
                    cond_embeddings_list = []
                    for i in range(16):
                        layer_ids = torch.full((batch_size,), i, device=device, dtype=torch.long)
                        moe_batch = MOEBatch(
                            input_ids=text_inputs.input_ids,
                            initialization_token_id=tokenizer(args.init_token).input_ids[1],
                            timesteps=t.repeat(batch_size),
                            task=args.task,
                            top_k=args.top_k,
                            top_k_general=args.top_k_general,
                            rescale=args.rescale,
                            layers=layer_ids,
                            placeholder_token=args.placeholder_token,
                            sampled_norm=tokenizer_l2_norms_mean_batch,
                        )
                        cond_embeddings = text_encoder(batch=moe_batch)[0].float()
                        uncond_embeddings = text_encoder(uncond_inputs.input_ids)[0].float()
                        full_embeddings = torch.cat([uncond_embeddings, cond_embeddings], dim=0)
                        cond_embeddings_list.append(full_embeddings)
                else:
                    moe_batch = MOEBatch(
                        input_ids=text_inputs.input_ids,
                        initialization_token_id=tokenizer(args.init_token).input_ids[1],
                        timesteps=t.repeat(batch_size),
                        task=args.task,
                        top_k=args.top_k,
                        top_k_general=args.top_k_general,
                        rescale=args.rescale,
                        placeholder_token=args.placeholder_token,
                        sampled_norm=tokenizer_l2_norms_mean_batch,
                    )
                    cond_embeddings = text_encoder(batch=moe_batch)[0].float()
                    uncond_embeddings = text_encoder(uncond_inputs.input_ids)[0].float()
                    full_embeddings = torch.cat([uncond_embeddings, cond_embeddings], dim=0)
                
                
                latent_model_input = torch.cat([latents, latents], dim=0)
                if args.layer_wise:
                    noise_pred = unet(latent_model_input, t, encoder_hidden_states_list=cond_embeddings_list).sample
                else:
                    noise_pred = unet(latent_model_input, t, encoder_hidden_states=full_embeddings).sample

                uncond_pred, cond_pred = noise_pred.chunk(2)
                noise_pred = uncond_pred + args.guidance_scale * (cond_pred - uncond_pred)
                latents = scheduler.step(noise_pred, t, latents)["prev_sample"]

            # Decode
            with torch.no_grad():
                images = vae.decode(latents / vae.config.scaling_factor).sample
                images = (images.clamp(-1, 1) + 1) / 2
                images = images.cpu().permute(0, 2, 3, 1).numpy()

            # Save
            for idx, prompt in enumerate(batch_prompts):
                out_dir = os.path.join(args.output_dir, f"{args.global_step}_{args.init_token}_lr{args.lr}{args.additional}_{mode}_prompts/seed_{seed}")
                os.makedirs(out_dir, exist_ok=True)
                Image.fromarray((images[idx] * 255).astype(np.uint8)).save(
                    os.path.join(out_dir, f"{prompt}.png")
                )

if __name__ == "__main__":
    args = parse_args()
    accelerator = Accelerator()
    device = accelerator.device

    token = args.placeholder_token.strip("<>")
    caption_attr = f"{token}_caption"
    caption = img_captions.captions_dict[caption_attr]

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = NETICLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    token_embeds = text_encoder.get_input_embeddings().weight.data
    caption_ids = tokenizer.encode(caption, add_special_tokens=False)
    caption_embedding = token_embeds[caption_ids].clone()
    init_vector = token_embeds[tokenizer.encode(args.init_token, add_special_tokens=False)[0]].clone()

    moe = NETIMixtureOfEmbedding(init_type=args.moe_init_type, init_vector=init_vector, top_k=args.top_k, init_caption=caption_embedding)
    moe.load_state_dict(torch.load(f"{args.embed_dir}/moe_weights_globalstep_{args.global_step}.pth"))
    text_encoder.text_model.embeddings.set_moe(moe)

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet_cls = PPlusUNet2DConditionModel if args.layer_wise else UNet2DConditionModel
    unet = unet_cls.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    if isinstance(text_encoder.text_model.embeddings, NETICLIPTextEmbeddings):
        text_encoder.text_model.embeddings.set_tokenizer(tokenizer)
        logger.info("Tokenizer set in NETICLIPTextEmbeddings for prompt logging.")
    else:
        logger.warning("Text encoder embeddings are not NETICLIPTextEmbeddings. Prompt logging might fail.")

    vae, unet, text_encoder, moe = accelerator.prepare(vae, unet, text_encoder, moe)
    for model in [vae, unet, text_encoder, moe]:
        model.eval()
        model.requires_grad_(False)

    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    accelerator = Accelerator()
    device = accelerator.device
    print(f"device: {args.device}")

    # load model
    # task : original, textual_inversion, p+, style_mixing
    if args.task == "original":
        print("loading the original pipeline")
        pipe = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.float16 if args.float16 else None).to(args.device)
    elif args.task == "textual_inversion":
        pipe = TextualInversionStableDiffusionPipeline.from_learned_embed(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            learned_embed_name_or_path=args.learned_embed_name_or_path,
            torch_dtype=torch.float16 if args.float16 else None,
        ).to(args.device)
    elif args.task == "p+":
        pipe = PPlusStableDiffusionPipeline.from_learned_embed(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            learned_embed_name_or_path=args.learned_embed_name_or_path,
            torch_dtype=torch.float16 if args.float16 else None,
        ).to(args.device)
    elif args.task == "style_mixing":
        pipe = PPlusStableDiffusionPipeline.from_learned_embed(
            pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4",
            learned_embed_name_or_path=[args.learned_embed_name_or_path, args.learned_embed_name_or_path_2],
            torch_dtype=torch.float16 if args.float16 else None,
            style_mixing_k_K=(5, 10),
        )
    pipe.scheduler = DPMSolverMultistepScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    
    # prepare pipeline's models (UNet, VAE, Text Encoder only)
    pipe.unet, pipe.text_encoder, pipe.vae = accelerator.prepare(
        pipe.unet, pipe.text_encoder, pipe.vae
    )
    if is_xformers_available():
        pipe.enable_xformers_memory_efficient_attention()
    print("loaded pipeline")

    with torch.no_grad():
        with accelerator.autocast():  # Automatic mixed precision (fp16 if enabled)
            if token in ["backpack_dog", "berry_bowl", "clock", "poop_emoji", "rc_car", "teapot" ]:
                prompts = prompt_list_1(unique_token=args.init_token)
                run_inference(prompts, mode="test_1")
            else:
                prompts = prompt_list_2(unique_token=args.init_token)
                run_inference(prompts, mode="test_2")
            
            # 2. imagenet_prompt_list로 생성
            prompts = imagenet_prompt_list(args.init_token)
            run_inference(prompts, mode="train")
