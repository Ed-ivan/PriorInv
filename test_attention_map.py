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
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as F

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
from utils.control_utils import load_512, make_controller,save_attention_map
# from P2P.SPDInv import SourcePromptDisentanglementInversion
from P2P.CFGInv_withloss import CFGInversion
from sklearn.decomposition import PCA
from ptp_utils import AttentionStore




# 可视化
def visualize_and_save_features_pca(feats_map, t, save_dir, layer_idx):
    B = len(feats_map)
    feats_map = feats_map.flatten(0, -2)
    feats_map = feats_map.cpu().numpy()
    pca = PCA(n_components=3)
    pca.fit(feats_map)
    feature_maps_pca = pca.transform(feats_map)  # N X 3
    feature_maps_pca = feature_maps_pca.reshape(B, -1, 3)  # B x (H * W) x 3
    for i, experiment in enumerate(feature_maps_pca):
        pca_img = feature_maps_pca[i]  # (H * W) x 3
        h = w = int(np.sqrt(pca_img.shape[0]))
        pca_img = pca_img.reshape(h, w, 3)
        pca_img_min = pca_img.min(axis=(0, 1))
        pca_img_max = pca_img.max(axis=(0, 1))
        pca_img = (pca_img - pca_img_min) / (pca_img_max - pca_img_min)
        pca_img = Image.fromarray((pca_img * 255).astype(np.uint8))
        pca_img.save(os.path.join(save_dir, f"{i}_time_{t}_layer_{layer_idx}.png"))

# from github masactrl issues pca feature maps 
def visualize_pca_results(con_image_pca, ucon_image_pca, delta_image_pca, t, output_dir, n_components):
    # 绘制热图
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(con_image_pca[:, :2], cmap='hot', interpolation='nearest')
    plt.title(f'Con Image PCA (t={t})')
    plt.xlabel('PC1, PC2')
    plt.ylabel('Pixel Index')
    plt.subplot(1, 3, 2)
    plt.imshow(ucon_image_pca[:, :2], cmap='hot', interpolation='nearest')
    plt.title(f'Ucon Image PCA (t={t})')
    plt.xlabel('PC1, PC2')
    plt.ylabel('Pixel Index')

    plt.subplot(1, 3, 3)
    plt.imshow(delta_image_pca[:, :2], cmap='hot', interpolation='nearest')
    plt.title(f'Delta Image PCA (t={t})')
    plt.xlabel('PC1, PC2')
    plt.ylabel('Pixel Index')

    plt.savefig(f"{output_dir}/pca_heatmap_{t}.png")
    plt.close()


@torch.no_grad()
def save_noise(noise_pred_con, noise_pred_ucon, t, output_dir='output_noise'):


'''
这个正则的代码应该怎么写呢？ 
先走一遍 存储里面的self-attention代码 
所以对于prompts 是 [a photo of cat # a photo of dog]? 

'''



class SelfAttentionStore(AttentionStore):
    @staticmethod
    def get_empty_store():
        return {"down_self": [], "mid_self": [], "up_self": []}
    #   使用的是 attend 环境别忘记了
        


def avg_attention_map(attn_dict):
    attn_size={}
    for key in attn_dict:
        attn_size[key]={}
        for item in attn_dict[key]:
            shape = item.shape[1]
            if shape not in attn_size[key]:
                attn_size[key][shape] = []
            attn_size[key][shape].append(item)    
    averaged_attn = {}
    for key in attn_size:
        averaged_attn[key] = {}
        for shape in attn_size[key]:
            attn_group = torch.stack(attn_size[key][shape], dim=0)
            averaged_attn[key][shape] = attn_group.mean(dim=0)
    return average_attention



def restore_self_attn(model,
        prompt: List[str],
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
    # 1、first loop to get self_attention 
    prompt =prompt[0]
    self_controller  = SelfAttentionStore()
    # make  self attention save controller 
    editing_p2p(prompt,self_controller,num_inference_steps,guidance_scale,generator,latent,uncond_embeddings,return_type,
    inference_stage,x_stars)
    # 想一哈 里面的 [src, tar] , [uncon,uncon]
    return self_controller


def editing_p2p_with_regular(image_path,
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
        noise_save_dir=None,
        **kwargs):
    #TODO：里面代码写的有点问题
    ref_controller = restore_self_attn(model,prompt,controller,num_inference_steps,guidance_scale,generator,latent)
    attn_dict = ref_controller.get_average_attention()
    avg_self_attn_dict = avg_attention_map(attn_dict)

    #直接 
    controller = make_controller(ldm_stable, prompts, is_replace_controller, cross_replace_steps, self_replace_steps,
                                 blend_word, eq_params, num_ddim_steps=num_of_ddim_steps)
    filename = image_path.split('/')[-1].replace(".jpg",".png")



    controller.set_ref_attn_dict(avg_self_attn_dict)

    images, _ = editing_p2p(ldm_stable, prompts, controller, latent=z_inverted_noise_code,
                            num_inference_steps=num_of_ddim_steps,
                            guidance_scale=guidance_scale,
                            uncond_embeddings=uncond_embeddings,
                            inversion_guidance=use_inversion_guidance, x_stars=x_stars, noise_save_dir=noise_save_dir)
    #形式就是 ： 
    # down_self dim [16,1024 , 1024]
    # down_self dim [16,256,256]
    # mid_self dim [16 ,64,64]
    # up_self dim[16 , 256,256]
    # up_self dim [16 , 1024 , 1024]

    
    



    assert output_dir is not None, "noise_save_dir can not be empty"
    os.makedirs(output_dir, exist_ok=True)
    resize = transforms.Resize((256, 256))
    noise_delta = noise_pred_con - noise_pred_ucon
    noise_pred_con_mean = torch.mean(noise_pred_con, dim=1, keepdim=True)
    noise_pred_ucon_mean = torch.mean(noise_pred_ucon, dim=1, keepdim=True)
    noise_delta_mean = torch.mean(noise_delta, dim=1, keepdim=True)
    noise_pred_con_mean = noise_pred_con_mean.expand(-1, 3, -1, -1)
    noise_pred_ucon_mean = noise_pred_ucon_mean.expand(-1, 3, -1, -1)
    noise_delta_mean = noise_delta_mean.expand(-1, 3, -1, -1)

    noise_pred_con_mean = resize(noise_pred_con_mean)
    noise_pred_ucon_mean = resize(noise_pred_ucon_mean)
    noise_delta_mean = resize(noise_delta_mean)
    con_image = noise_pred_con_mean[0].cpu().numpy().transpose(1, 2, 0)
    ucon_image = noise_pred_ucon_mean[0].cpu().numpy().transpose(1, 2, 0)
    delta_image = noise_delta_mean[0].cpu().numpy().transpose(1, 2, 0)

    con_image = (con_image - con_image.min()) / (con_image.max() - con_image.min())
    ucon_image = (ucon_image - ucon_image.min()) / (ucon_image.max() - ucon_image.min())
    delta_image = (delta_image - delta_image.min()) / (delta_image.max() - delta_image.min())

    #app PCA 
    pca = PCA(n_components=4)
    con_image_pca = pca.fit_transform(con_image.reshape(-1, con_image.shape[2])).reshape(con_image.shape)
    ucon_image_pca = pca.fit_transform(ucon_image.reshape(-1, ucon_image.shape[2])).reshape(ucon_image.shape)
    delta_image_pca = pca.fit_transform(delta_image.reshape(-1, delta_image.shape[2])).reshape(delta_image.shape)

    con_image_pca = (con_image_pca * 255).astype(np.uint8)
    ucon_image_pca = (ucon_image_pca * 255).astype(np.uint8)
    delta_image_pca = (delta_image_pca * 255).astype(np.uint8)

    # save images
    Image.fromarray(con_image_pca).save(f"{output_dir}/noise_pred_con_pca_{t}.png")
    Image.fromarray(ucon_image_pca).save(f"{output_dir}/noise_pred_ucon_pca_{t}.png")
    Image.fromarray(delta_image_pca).save(f"{output_dir}/noise_delta_pca_{t}.png")


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
    assert noise_save_dir is not None ,"noise_save_dir can not be empty"
    @torch.no_grad()
    def capture_noise_test(noise_pred_con,noise_pred_ucon,t):
        save_noise(noise_pred_con,noise_pred_ucon,t,noise_save_dir)
    


    
    with torch.no_grad():
        for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:], total=num_inference_steps)):
            if uncond_embeddings_ is None:
                context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
            else:
                context = torch.cat([uncond_embeddings_, text_embeddings])
            # TODO：应该在里面 重新写一下吧 ？  
            latents = ptp_utils.diffusion_step(model, controller, latents, context, t, guidance_scale,
                                               low_resource=False,
                                               inference_stage=inference_stage, x_stars=x_stars,i=i, capture_noise=capture_noise_test,**kwargs)
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
        noise_save_dir=None,
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
                            inversion_guidance=use_inversion_guidance, x_stars=x_stars, noise_save_dir=noise_save_dir)

    
    #save_attention_map(ldm_stable.tokenizer, controller, res=16,prompts=prompts,from_where=["up", "down"],filename=f"{output_dir}"+"attention_map")
    # attn = controller.get_average_attention()
    # for key in controller.attention_store:
    #     for item in controller.attention_store[key]:
    #         print(f"here {key} dim is ",item.size())
    # 所以得到了 attention_map 之后 ，这里只是简单的取平均，取平均之后呢？ 确实啊，这里面的代码其实挺不好写的
    # 还有一种写法就是将self_controller作为一个参数，每次进行append的时候直接 mix_up？ 参数应该怎么给？
    # 因为是先cross-attention才是self ,我看看 







    

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
        default="output_res_1125",
        help="Save editing results",
    )
    parser.add_argument(
        "--noise_save_dir",
        type=str,
        default="output_noise_save_1125",
        help="Save noise_pred results",
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
    params['noise_save_dir'] =args.noise_save_dir
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


# down_cross dim  [16, 1024, 77]
# down_cross dim [16,256,77]
# mid_cross dim [16,64,77]
# up_cross dim [16,256,77]
# up_cross dim [16,1024,77]


# down_self dim [16,1024 , 1024]
# down_self dim [16,256,256]
# mid_self dim [16 ,64,64]
# up_self dim[16 , 256,256]
# up_self dim [16 , 1024 , 1024]


#NOTE: this function is not used!
def regulize_cross_attention(attn_dict):
    '''
        down_cross dim  [16, 1024, 77]
        down_cross dim [16,256,77]
        mid_cross dim [16,64,77]
        up_cross dim [16,256,77]
        up_cross dim [16,1024,77]
        down_self dim [16,1024 , 1024]
        down_self dim [16,256,256]
        mid_self dim [16 ,64,64]
        up_self dim[16 , 256,256]
        up_self dim [16 , 1024 , 1024]
    '''
    reguliazed_cross_attn_dict={}
    self_list = ["down_self","up_self","mid_self"]
    for key in attn_dict:
        if key in self_list:
            continue
        reguliazed_cross_attn_dict[key]={}
        for item in attn_dict[key]:
            shape = item.shape[1]
            if shape not in reguliazed_cross_attn_dict[key]:
                reguliazed_cross_attn_dict[key][shape] = []
            self_attn = attn_dict[key.replace("cross","self")][shape] 
            reguliazed_cross_attn_dict[key][shape].append(torch.einsum('bjc,bji->bic',self_attn,attn_dict[key][shape]))
            # 考虑其他位置的cross-attention-map
    return reguliazed_cross_attn_dict


#提出这种region-based-dynamic-guidance 是为了解决这种local edit的， generate mask 的时候需要原始的index ，看下>>

def generate_mask(attn_dict, idx: int, res=64):
    mask = None
    for key in attn_dict:
        for item in attn_dict[key]:
            shape = item.shape[1]
            
            chunked_item = item.chunk(2, dim=0)[0]
            
            chunked_item = chunked_item.view(8, -1, shape, 77)
            
            averaged_item = chunked_item.mean(dim=0)
            
            # 将结果调整为 res x res 的维度
            resized_item = F.interpolate(averaged_item.unsqueeze(0), size=(res, res), mode='bilinear').squeeze(0)
            if mask is None:
                mask = resized_item
            else:
                mask += resized_item
    
    mask /= len(attn_dict)
    
    return mask[:,:,idx]

