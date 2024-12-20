#引入 P(x)来约束  z_lanent 
import time
# %%
from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
from P2P import ptp_utils
from P2P import seq_aligner
import shutil
from torch.optim.adam import Adam
from PIL import Image
import os
from P2P.scheduler_dev import DDIMSchedulerDev
import torch.nn.functional as F
from datetime import datetime

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h - bottom, left:w - right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


class Inversion:
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(
            timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self,latents,t,context):
        #但是这么修改的话， 直接就会出现了错误  。 应该怎么修改？ 
        latents_input = torch.cat([latents] * 2)
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_cfg = noise_pred_uncond + self.scale *(noise_prediction_text-noise_pred_uncond)
        return noise_cfg



    @torch.no_grad()     
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def SPD_loop(self, latent):
        raise NotImplementedError

    @property
    def scheduler(self):
        return self.model.scheduler

    def SPD_inversion(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)

        SPDInv_latents = self.SPD_loop(latent)

        return image_rec, SPDInv_latents, latent

    def invert(self, image_path: str, prompt: str, offsets=(0, 0, 0, 0), npi_interp=0.0, verbose=False):
        self.init_prompt(prompt)
        ptp_utils.register_attention_control(self.model, None)

        image_gt = load_512(image_path, *offsets)

        image_rec, ddim_latents, image_rec_latent = self.SPD_inversion(image_gt)
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        if npi_interp > 0.0:
            cond_embeddings = ptp_utils.slerp_tensor(npi_interp, cond_embeddings, uncond_embeddings)
        uncond_embeddings = [cond_embeddings] * self.num_ddim_steps
        return (image_gt, image_rec, image_rec_latent), ddim_latents, uncond_embeddings
    

    # help function : compute posterior_mean_variable
    def posterior_mean_variable(self,timestep: int, latent_init: Union[torch.FloatTensor, np.ndarray]):
        
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        # tensor,scalor
        return  alpha_prod_t**0.5*latent_init ,  1-alpha_prod_t

    def log_prob_regulation(self,latent: Union[torch.FloatTensor, np.ndarray],posterior_mean, posterior_variable):
        
        '''
        得到里面的 p(x_{t}|x_{0})分布,计算得到的x_{t}的概率，但是由于本身 x_t 的维度非常大,所以还不能直接这么写 
        logq(x_{t}|x_{0}) = sum(log(x_{tj}|x_{oj})) (按照每个都是独立分布进行处理的)
        '''
        #log_pz = 0.5 * torch.sum(torch.log(2 * torch.pi * posterior_variable*2)) + torch.sum((latent - posterior_mean)**2 / (2 * posterior_variable**2))
        #log_pz =  torch.mean((latent - posterior_mean)**2 / (2 * posterior_variable))
        log_pz =  torch.mean((latent - posterior_mean)**2)
        return log_pz

    #TODO:  ProxEdit_Improving_Tuning-Free_Real_Image_Editing_With_Proximal_Guidance  this fn is not used !
    def proximal_constants(self,prox,noise_prediction_text,noise_pred_uncond,quantile): 

        '''
        主要是为了  限制 使用 CFG_inversion 时候 embeddings_un + scale(embddings_text -embeddings_un) 
        减少一下 得到的 noise,还需要将其中的函数进行某种程度的操作??
        '''
        if prox == 'l1':
            score_delta = noise_prediction_text - noise_pred_uncond
            if quantile > 0:
                threshold = score_delta.abs().quantile(quantile)
            else:
                threshold = -quantile  # if quantile is negative, use it as a fixed threshold
            score_delta -= score_delta.clamp(-threshold, threshold)
            score_delta = torch.where(score_delta > 0, score_delta-threshold, score_delta)
            score_delta = torch.where(score_delta < 0, score_delta+threshold, score_delta)
        pass

    def __init__(self, model, K_round=25, num_ddim_steps=50, learning_rate=0.001, delta_threshold=5e-6,
                 enable_threshold=True,scale =1.0,prior_lambda=0.00045):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False)
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(num_ddim_steps)
        self.prompt = None
        self.context = None
        self.opt_round = K_round
        self.num_ddim_steps = num_ddim_steps
        self.lr = learning_rate
        self.threshold = delta_threshold
        self.enable_threshold = enable_threshold
        self.scale = scale
        self.prior_lambda = prior_lambda


class CFGInversion(Inversion):
    @torch.no_grad()
    def SPD_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        latent_init = latent.clone().detach()
        for i in range(self.num_ddim_steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, self.context)
            # TODO: 那么就是需要修改这个地方了 还得得到这个uncond的  
            latent_ztm1 = latent.clone().detach()
            latent = self.next_step(noise_pred, t, latent_ztm1)
            ################ below code is from  SPDInv optimization steps #################

            # below is modified 
            prior_mean ,prior_variance = self.posterior_mean_variable(t,latent_init)
            prior_mean.requires_grad = False
            prior_variance.requires_grad=False
            # before is added! 
            optimal_latent = latent.clone().detach()
            
            optimal_latent.requires_grad = True
            #optimizer = torch.optim.SGD([optimal_latent], lr=self.lr, momentum=0.5, nesterov=True)
            optimizer = torch.optim.AdamW([optimal_latent], lr=self.lr)
            for rid in range(self.opt_round):
                with torch.enable_grad():
                    
                    optimizer.zero_grad()
                    noise_pred = self.get_noise_pred_single(optimal_latent, t, self.context)
                    # [1,4,64,64]
                    pred_latent = self.next_step(noise_pred, t, latent_ztm1)
                    
                    loss = F.mse_loss(optimal_latent, pred_latent)
                    
                    prior_loss = self.prior_lambda * self.log_prob_regulation(optimal_latent,prior_mean,prior_variance)
                    total_loss = loss+ prior_loss
                    #print("prior_loss is : ",prior_loss)
                    total_loss.backward()
                    optimizer.step()

                    if self.enable_threshold and loss < self.threshold:
                        break
            
            ############### End SPDInv optimization ###################

            latent = optimal_latent.clone().detach()
            latent.requires_grad = False
            all_latent.append(latent)
        return all_latent

    # help function : compute posterior_mean_variable
    def posterior_mean_variable(self,timestep: int, latent_init: Union[torch.FloatTensor, np.ndarray]):
        
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        # tensor,scalor
        return  alpha_prod_t**0.5*latent_init ,  1-alpha_prod_t

    def log_prob_regulation(self,latent: Union[torch.FloatTensor, np.ndarray],posterior_mean, posterior_variable):
        
        '''
        得到里面的 p(x_{t}|x_{0})分布,计算得到的x_{t}的概率，但是由于本身 x_t 的维度非常大,所以还不能直接这么写 
        logq(x_{t}|x_{0}) = sum(log(x_{tj}|x_{oj})) (按照每个都是独立分布进行处理的)
        '''
        #log_pz = 0.5 * torch.sum(torch.log(2 * torch.pi * posterior_variable*2)) + torch.sum((latent - posterior_mean)**2 / (2 * posterior_variable**2))
        #log_pz =  torch.mean((latent - posterior_mean)**2 / (2 * posterior_variable))
        log_pz =  torch.mean((latent - posterior_mean)**2)
        return log_pz

    #TODO:  ProxEdit_Improving_Tuning-Free_Real_Image_Editing_With_Proximal_Guidance  this fn is not used !
    def proximal_constants(self,prox,noise_prediction_text,noise_pred_uncond,quantile): 

        '''
        主要是为了  限制 使用 CFG_inversion 时候 embeddings_un + scale(embddings_text -embeddings_un) 
        减少一下 得到的 noise,还需要将其中的函数进行某种程度的操作??
        '''
        if prox == 'l1':
            score_delta = noise_prediction_text - noise_pred_uncond
            if quantile > 0:
                threshold = score_delta.abs().quantile(quantile)
            else:
                threshold = -quantile  # if quantile is negative, use it as a fixed threshold
            score_delta -= score_delta.clamp(-threshold, threshold)
            score_delta = torch.where(score_delta > 0, score_delta-threshold, score_delta)
            score_delta = torch.where(score_delta < 0, score_delta+threshold, score_delta)
        pass

    def __init__(self, model, K_round=25, num_ddim_steps=50, learning_rate=0.001, delta_threshold=5e-6,
                 enable_threshold=True,scale =1.0,prior_lambda=0.00045):
        super(CFGInversion,self).__init__(model,K_round,num_ddim_steps,learning_rate,delta_threshold,
        enable_threshold,scale,prior_lambda)

#this idea is from  020 
class CFGInversionWithRegular(Inversion):
    @torch.no_grad()
    def SPD_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        latent_init = latent.clone().detach()
        for i in range(self.num_ddim_steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, self.context)
            # TODO: 那么就是需要修改这个地方了 还得得到这个uncond的  
            latent_ztm1 = latent.clone().detach()
            latent = self.next_step(noise_pred, t, latent_ztm1)
            ################ below code is from  SPDInv optimization steps #################

            # below is modified 
            prior_mean ,prior_variance = self.posterior_mean_variable(t,latent_init)
            prior_mean.requires_grad = False
            prior_variance.requires_grad=False
            # before is added! 
            optimal_latent = latent.clone().detach()
            
            optimal_latent.requires_grad = True
            #optimizer = torch.optim.SGD([optimal_latent], lr=self.lr, momentum=0.5, nesterov=True)
            optimizer = torch.optim.AdamW([optimal_latent], lr=self.lr)
            for rid in range(self.opt_round):
                with torch.enable_grad():
                    
                    optimizer.zero_grad()
                    noise_pred = self.get_noise_pred_single(optimal_latent, t, self.context)
                    # [1,4,64,64]
                     # regularization of the noise prediction 
                     # https://github.com/pix2pixzero/pix2pix-zero/blob/main/src/utils/ddim_inv.py
                    e_t = noise_pred
                    for _outer in range(self.num_reg_steps):
                        if self.lambda_ac>0:
                            for _inner in range(self.num_ac_rolls):
                                _var = torch.autograd.Variable(e_t.detach().clone(), requires_grad=True)
                                l_ac = self.auto_corr_loss(_var)
                                l_ac.backward()
                                _grad = _var.grad.detach()/self.num_ac_rolls
                                e_t = e_t - self.lambda_ac*_grad
                        if self.lambda_kl>0:
                            _var = torch.autograd.Variable(e_t.detach().clone(), requires_grad=True)
                            l_kld = self.kl_divergence(_var)
                            l_kld.backward()
                            _grad = _var.grad.detach()
                            e_t = e_t - self.lambda_kl*_grad
                        e_t = e_t.detach()
                    noise_pred = e_t
                    pred_latent = self.next_step(noise_pred, t, latent_ztm1)
                    loss = F.mse_loss(optimal_latent, pred_latent)
                    
                    loss.backward()
                    optimizer.step()

                    if self.enable_threshold and loss < self.threshold:
                        break
            
            ############### End SPDInv optimization ###################

            latent = optimal_latent.clone().detach()
            latent.requires_grad = False
            all_latent.append(latent)
        return all_latent

    def __init__(self, model, K_round=25, num_ddim_steps=50, learning_rate=0.001, delta_threshold=5e-6,
                 enable_threshold=True,scale =1.0,prior_lambda=0.00045,
                 lambda_ac=0.00045,num_ac_rolls=5,lambda_kl=5,num_reg_steps=5):
        
        super(CFGInversionWithRegular,self).__init__(model,K_round,num_ddim_steps,learning_rate,delta_threshold,
        enable_threshold,scale,prior_lambda)
        self.num_ac_rolls= num_ac_rolls
        self.lambda_ac= lambda_ac
        self.lambda_kl = lambda_kl
        self.num_reg_steps = num_reg_steps
        

# which is about src and target 
class BlendInversion(CFGInversion):
    def invert(self, image_path: str, prompt_src: str, prompt_tar:str,offsets=(0, 0, 0, 0), npi_interp=0.0, verbose=False):
        self.init_prompt(prompt_src,prompt_tar)
        ptp_utils.register_attention_control(self.model, None)

        image_gt = load_512(image_path, *offsets)

        image_rec, ddim_latents, image_rec_latent = self.SPD_inversion(image_gt)
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        if npi_interp > 0.0:
            cond_embeddings = ptp_utils.slerp_tensor(npi_interp, cond_embeddings, uncond_embeddings)
        uncond_embeddings = [cond_embeddings] * self.num_ddim_steps
        return (image_gt, image_rec, image_rec_latent), ddim_latents, uncond_embeddings
    


    @torch.no_grad()
    def init_prompt(self, prompt: str,prompt_tar:str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_tar = self.model.tokenizer(
            [prompt_tar],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        text_embeddings_tar = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, self.blend_ratio*text_embeddings +(1-self.blend_ratio) * text_embeddings_tar])
        self.prompt = prompt

    def __init__(self, model, K_round=25, num_ddim_steps=50, learning_rate=0.001, delta_threshold=5e-6,
                 enable_threshold=True,scale =1.0,prior_lambda=0.00045,blend_ratio=0.2):
        super(CFGInversion,self).__init__(model,K_round,num_ddim_steps,learning_rate,delta_threshold,
        enable_threshold,scale,prior_lambda)
        self.blend_ratio= blend_ratio
        #TODO: 这个应该是 随着时间变化的啊， 我真的是吐了啊，没事先不要慌张, 我记得有一个来着是怎么来做的呢？

# 既然 反演过程中应该是考虑 编辑文本的,如何考虑呢 ？（1）最简单的方法   text embeddings空间中进行插值   
# (2) delta的方法 使用 a*e() + (1-a)*e( ) 这种方法  实现的时候用的比较 tricky的实现方法 直接将 uncond置为了prompt_src 
class DeltaInversion(CFGInversion):
    
    @torch.no_grad()
    def init_prompt(self, prompt_src: str,prompt_tar:str):
        uncond_input = self.model.tokenizer(
            [prompt_src], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]

        text_input_tar = self.model.tokenizer(
            [prompt_tar],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt_tar



    def invert(self, image_path: str, prompt_src: str,  prompt_tar:str,offsets=(0, 0, 0, 0), npi_interp=0.0, verbose=False):
        self.init_prompt(prompt_src,prompt_tar)
        ptp_utils.register_attention_control(self.model, None)

        image_gt = load_512(image_path, *offsets)

        image_rec, ddim_latents, image_rec_latent = self.SPD_inversion(image_gt)
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        if npi_interp > 0.0:
            cond_embeddings = ptp_utils.slerp_tensor(npi_interp, cond_embeddings, uncond_embeddings)
        uncond_embeddings = [cond_embeddings] * self.num_ddim_steps
        return (image_gt, image_rec, image_rec_latent), ddim_latents, uncond_embeddings
    
