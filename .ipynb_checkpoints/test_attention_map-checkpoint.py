'''
author:jin 
data: 2024.11.22
MIT 协议
'''

'''
for
'''
from typing import Optional, Union, List
from tqdm import tqdm
import torch
from diffusers import StableDiffusionPipeline
import numpy as np

from P2P import ptp_utils
from PIL import Image
import os
from P2P.scheduler_dev import DDIMSchedulerDev
import argparse
from torchvision import transforms
import torch.nn.functional as F

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
from utils.control_utils import load_512, make_controller,save_attention_map
# from P2P.SPDInv import SourcePromptDisentanglementInversion
from P2P.CFGInv_withloss import CFGInversion


@torch.no_grad
def capture_noise(noise_pred_con ,noise_pred_ucon):
    #save_noise(noise_pred_con,noise_pred_ucon,noise_save_dir)
    pass 


@torch.no_grad()
def editing_p2p(
        model,
        prompt: List[str],
        controller,
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = 7.5,
        generator: Optional[torch.Generator] = None,
        latent: Optional[torch.FloatTensor] = None,
        uncond_embeddings=None,
        return_type='image',
        inference_stage=True,
        x_stars=None,
        noise_save_dir=None,
        **kwargs,
):
    batch_size = len(prompt)
    ptp_utils.register_attention_control(model, controller)
    # 应该是在 进行了注册 
    height = width = 512

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    # [2,77.768] 
    max_length = text_input.input_ids.shape[-1]


    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)
    start_time = num_inference_steps
    model.scheduler.set_timesteps(num_inference_steps)
    
    noise_ucon_list =[]
    noise_con_list =[]
    # callback func for  save noise_delta 


    
    with torch.no_grad():
        for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:], total=num_inference_steps)):
            if uncond_embeddings_ is None:
                context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
            else:
                context = torch.cat([uncond_embeddings_, text_embeddings])
            # TODO：应该在里面 重新写一下吧 ？  
            latents = ptp_utils.diffusion_step(model, controller, latents, context, t, guidance_scale,
                                               low_resource=False,
                                               inference_stage=inference_stage, x_stars=x_stars,i=i,**kwargs)
    if return_type == 'image':
        image = ptp_utils.latent2image(model.vae, latents)
        
    else:
        image = latents
        
    return image, latent


@torch.no_grad()
def show_attention_map(
        image_path,
        prompt_src,
        prompt_tar,
        output_dir='output',
        guidance_scale=5.0,
        npi_interp=0,
        cross_replace_steps=0.8,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        offsets=(0, 0, 0, 0),
        is_replace_controller=False,
        use_inversion_guidance=True,
        K_round=25,
        num_of_ddim_steps=50,
        learning_rate=0.001,
        delta_threshold=5e-6,
        enable_threshold=True,
        **kwargs
):
    os.makedirs(output_dir, exist_ok=True)
    sample_count = len(os.listdir(output_dir))
    scheduler = DDIMSchedulerDev(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                 set_alpha_to_one=False)
    ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=scheduler).to(
        device)
   
    
    SPD_inversion = CFGInversion(ldm_stable, K_round=K_round, num_ddim_steps=num_of_ddim_steps,
                                                         learning_rate=learning_rate, delta_threshold=delta_threshold,
                                                         enable_threshold=enable_threshold)
    (image_gt, image_enc, image_enc_latent), x_stars, uncond_embeddings = SPD_inversion.invert(
        image_path, prompt_src, offsets=offsets, npi_interp=npi_interp, verbose=True)
    tokenizer=SPD_inversion.model.tokenizer
    z_inverted_noise_code = x_stars[-1]
    torch.cuda.empty_cache()
    del SPD_inversion

    ########## edit ##########
    prompts = [prompt_src, prompt_tar]
    cross_replace_steps = {'default_': cross_replace_steps, }
    if isinstance(blend_word, str):
        s1, s2 = blend_word.split(",")
        blend_word = (((s1,), (
            s2,)))  # for local edit. If it is not local yet - use only the source object: blend_word = ((('cat',), ("cat",))).
    if isinstance(eq_params, str):
        s1, s2 = eq_params.split(",")
        eq_params = {"words": (s1,), "values": (float(s2),)}  # amplify attention to the word "tiger" by *2
    controller = make_controller(ldm_stable, prompts, is_replace_controller, cross_replace_steps, self_replace_steps,
                                 blend_word, eq_params, num_ddim_steps=num_of_ddim_steps)
    filename = image_path.split('/')[-1].replace(".jpg",".png")

    images, _ = editing_p2p(ldm_stable, prompts, controller, latent=z_inverted_noise_code,
                            num_inference_steps=num_of_ddim_steps,
                            guidance_scale=guidance_scale,
                            uncond_embeddings=uncond_embeddings,
                            inversion_guidance=use_inversion_guidance, x_stars=x_stars, )
    attn = controller.get_average_attention()
    for key in controller.attention_store:
        for item in controller.attention_store[key]:
            print(f"here {key} dim is ",item.size())


    

def parse_args():
    parser = argparse.ArgumentParser(description="Input your image and editing prompt.")
    parser.add_argument(
        "--input",
        type=str,
        default="images/000000000001.jpg",
        # /home/user/jin/SPDInv/images/gnochi_mirror.jpeg
        # images/000000000008.jpg
        # images/000000000138.jpg
        # required=True,
        help="Image path",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="a round cake with orange frosting on a wooden plate",
        # required=True,
        # a round cake with orange frosting on a wooden plate A cat sitting next to a mirror
        # a Golden Retriever standing on the groud
        help="Source prompt",
    )
    parser.add_argument(
        "--target",
        type=str,
        default= "a square cake with orange frosting on a wooden plate",
        #"a Golden Retriever",
        # a silver cat  sculpture standing on the groud
        # required=True,
        # a square cake with orange frosting on a wooden plate
        help="Target prompt",
    )
    parser.add_argument(
        "--blended_word",
        type=str,
        default="cake cake",
        help="Blended word needed for P2P",
    )
    parser.add_argument(
        "--K_round",
        type=int,
        default=25,
        help="Optimization Round",
    )
    parser.add_argument(
        "--num_of_ddim_steps",
        type=int,
        default=50,
        help="Blended word needed for P2P",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--delta_threshold",
        type=float,
        default=5e-6,
        help="Delta threshold",
    )
    parser.add_argument(
        "--enable_threshold",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--eq_params",
        type=float,
        default=2.,
        help="Eq parameter weight for P2P",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_res_1113",
        help="Save editing results",
    )

    parser.add_argument(
        "--show_cross",
        type=bool,
        default=True,
        # required=True,
        # a round cake with orange frosting on a wooden plate A cat sitting next to a mirror
        # a Golden Retriever standing on the groud
        help="Show cross_attention_map or self_attention_map",
    )
    args = parser.parse_args()
    return args

# 里面的具体编辑还得看下人家 direactinversion 的  


if __name__ == "__main__":
    args = parse_args()
    params = {}
    params['guidance_scale'] = args.guidance_scale
    params['blend_word'] = (((args.blended_word.split(" ")[0],), (args.blended_word.split(" ")[1],)))
    params['eq_params'] = {"words": (args.blended_word.split(" ")[1],), "values": (args.eq_params,)}
    params['K_round'] = args.K_round
    params['num_of_ddim_steps'] = args.num_of_ddim_steps
    params['learning_rate'] = args.learning_rate
    params['enable_threshold'] = args.enable_threshold
    params['delta_threshold'] = args.delta_threshold

    params['prompt_src'] = args.source
    params['prompt_tar'] = args.target 
    params['output_dir'] = args.output
    params['image_path'] = args.input
    #P2P_inversion_and_recontruction(**params)
    show_attention_map(**params)
    #  HF_ENDPOINT=https://hf-mirror.com python test_attention_map.py  






# here down_cross dim is  torch.Size([16, 1024, 77])
# here down_cross dim is  torch.Size([16, 1024, 77])
# here down_cross dim is  torch.Size([16, 256, 77])
# here down_cross dim is  torch.Size([16, 256, 77])
# here mid_cross dim is  torch.Size([16, 64, 77])
# here up_cross dim is  torch.Size([16, 256, 77])
# here up_cross dim is  torch.Size([16, 256, 77])
# here up_cross dim is  torch.Size([16, 256, 77])
# here up_cross dim is  torch.Size([16, 1024, 77])
# here up_cross dim is  torch.Size([16, 1024, 77])
# here up_cross dim is  torch.Size([16, 1024, 77])


# here down_self dim is  torch.Size([16, 1024, 1024])
# here down_self dim is  torch.Size([16, 1024, 1024])
# here down_self dim is  torch.Size([16, 256, 256])
# here down_self dim is  torch.Size([16, 256, 256])
# here mid_self dim is  torch.Size([16, 64, 64])
# here up_self dim is  torch.Size([16, 256, 256])
# here up_self dim is  torch.Size([16, 256, 256])
# here up_self dim is  torch.Size([16, 256, 256])
# here up_self dim is  torch.Size([16, 1024, 1024])
# here up_self dim is  torch.Size([16, 1024, 1024])