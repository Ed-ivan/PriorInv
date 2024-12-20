from typing import Optional, Union, Tuple, List, Dict
import torch
import torch.nn.functional as nnf
import numpy as np
import abc
from P2P import ptp_utils, seq_aligner
from PIL import Image

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

LOW_RESOURCE = False
# NUM_DDIM_STEPS = 50
MAX_NUM_WORDS = 77
LATENT_SIZE = (64, 64)


class LocalBlend:
    def get_mask(self, maps, alpha, use_pool):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        #  是针对于整个的prompts的
        if use_pool:
            maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=LATENT_SIZE)
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1 - int(use_pool)])
        mask = mask[:1] + mask
        return mask

    def __call__(self, x_t, attention_store):
        self.counter += 1
        #TODO: 这里面应该可以修改一下的吧 
        if self.counter > self.start_blend:
            down_cross_attn = attention_store["down_cross"][2:4]
            down_self_attn = attention_store["down_self"][2:4]
            up_cross_attn = attention_store["up_cross"][:3]
            up_self_attn = attention_store["up_self"][:3]
            cross_attns_down =[]
            cross_attns_up =[]
            # for down
            for cross_attn, self_attn in zip(down_cross_attn, down_self_attn):
                result = torch.einsum('bjc,bji->bic', cross_attn, self_attn)
                cross_attns_down.append(result)
            # for up 
            for cross_attn, self_attn in zip(up_cross_attn, up_self_attn):
                result = torch.einsum('bjc,bji->bic', cross_attn, self_attn)
                cross_attns_up.append(result)

            maps = cross_attns_down + cross_attns_up

            maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
            maps = torch.cat(maps, dim=1)
            mask = self.get_mask(maps, self.alpha_layers, True)
            # maps size is : ([2, 40, 1, 16, 16, 77])
            #mask size is [2,4,64,64] 
            if self.substruct_layers is not None:
                maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
                mask = mask * maps_sub
            mask = mask.float()
            
            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t

    def __init__(self, prompts: List[str], words: [List[List[str]]], substruct_words=None, start_blend=0.2, th=(.3, .3),
                 tokenizer=None, num_ddim_steps=20):
        alpha_layers = torch.zeros(len(prompts), 1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1

        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts), 1, 1, 1, 1, MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * num_ddim_steps)
        self.counter = 0
        self.th = th 


class EmptyControl:

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead        
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
        return average_attention
    

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.ref_attn_dict = None 
        
    #NOTE: 添加了set 属性
    def set_ref_attn_dict(self,ref_attn_dict=None):
        if ref_attn_dict is not None:
            self.ref_attn_dict = ref_attn_dict
        


#NOTE:  self defined function to restore self attention ,which is used to regularize the cross attention map
class SelfAttentionStore(AttentionStore):
    @staticmethod
    def get_empty_store():
        return {"down_self": [], "mid_self": [], "up_self": []}
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            if not is_cross:
                self.step_store[key].append(attn)
        return attn

    def avg_attention_map(self):
        attn_dict= self.get_average_attention()
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
        return averaged_attn


    def __init__(self):
        super(SelfAttentionStore, self).__init__()
        




class AttentionControlEdit(AttentionStore, abc.ABC):

    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t

    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 32 ** 2:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            # 这代码 写的可以 ，将两者存到 controller中，
            attn_base, attn_repalce = attn[0], attn[1:]
            attn_base_copy= attn_base.detach().clone()
            if is_cross:
                ###################### add code of cross attention regularization  ########################
                key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
                if self.ref_attn_dict is not None:
                    #TODO： 应该怎么使用 self_attention进行 regular? 并且
                    if attn_base.shape[1] <= 32**2:
                        item = self.ref_attn_dict[key.replace('cross','self')][attn_base.shape[1]]
                        assert attn_base.shape[0] == item[:attn_base.shape[0]].shape[0] ,"ref dim is not same with edit"
                        attn_base_copy = attn_base.clone()
                        # 在副本上进行修改 , 我靠绝了啊 ， attn 不能动的
                        attn_base_copy.copy_(torch.einsum('bjc,bji->bic', attn_base_copy, item[:attn_base.shape[0]]))
                    # 8 1024 , 77 ? 
                ###################### add code of cross attention regularization  ########################
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base_copy, attn_repalce) * alpha_words + (
                            1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new

            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend], tokenizer=None):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps,
                                                                            tokenizer).to(device)
        # size is 51 1 1 1 77 
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend


class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)

        #self.mapper 是个什么玩应？ ， 
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None, tokenizer=None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)
        # ([1, 77]) 


class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None, tokenizer=None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        # [1,77]
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])
        
class AttentionReweight(AttentionControlEdit):
    @torch.no_grad()
    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                 local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps,
                                                local_blend)
        self.equalizer = equalizer.to(device)
        # 【1，77】 跟之后的做广播
        self.prev_controller = controller


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
Tuple[float, ...]], tokenizer=None):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)

    for word, val in zip(word_select, values):
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer


def make_controller(pipeline, prompts: List[str], is_replace_controller: bool, cross_replace_steps: Dict[str, float],
                    self_replace_steps: float, blend_words=None, equilizer_params=None,
                    num_ddim_steps=20) -> AttentionControlEdit:
    if blend_words is None:
        lb = None
    else:
        lb = LocalBlend(prompts, blend_words, tokenizer=pipeline.tokenizer, num_ddim_steps=num_ddim_steps)

    if is_replace_controller:
        controller = AttentionReplace(prompts, num_ddim_steps, cross_replace_steps=cross_replace_steps,
                                      self_replace_steps=self_replace_steps, local_blend=lb,
                                      tokenizer=pipeline.tokenizer)
    else:
        controller = AttentionRefine(prompts, num_ddim_steps, cross_replace_steps=cross_replace_steps,
                                     self_replace_steps=self_replace_steps, local_blend=lb,
                                     tokenizer=pipeline.tokenizer)
    if equilizer_params is not None:
        eq = get_equalizer(prompts[1], equilizer_params["words"], equilizer_params["values"],
                           tokenizer=pipeline.tokenizer)
        # eq size [1,77] 
        # 就是将对应的词的attention-map进行放大 
        controller = AttentionReweight(prompts, num_ddim_steps, cross_replace_steps=cross_replace_steps,
                                       self_replace_steps=self_replace_steps, equalizer=eq, local_blend=lb,
                                       controller=controller)
    return controller


def read_image(image_path):
    try:

        image = Image.open(image_path)
        return np.array(image)
    except Exception as e:
        print("Error",e)
        return None
    


def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int,prompts):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels: 
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()




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

def load_image(image_path, device):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = T.Resize(512)(image)
    image = T.CenterCrop(512)(image)
    image = image.to(device)
    return image



'''
  File "/root/autodl-tmp/CFGInv/utils/control_utils.py", line 317, in aggregate_attention
    cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
RuntimeError: shape '[51, -1, 16, 16, 77]' is invalid for input of size 315392
'''

def save_attention_map(tokenizer,attention_store: AttentionStore, res: int,prompts:dict,from_where: List[str], filename,select: int = 0,is_cross: bool = True) :
   
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select,prompts)
    images = []
    if not is_cross:
        max_com=10
        attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
        u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
        images = []
        for i in range(max_com):
            image = vh[i].reshape(res, res)
            image = image - image.min()
            image = 255 * image / image.max()
            image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
            image = Image.fromarray(image).resize((256, 256))
            image = np.array(image)
            images.append(image)
    else :
        for i in range(len(tokens)):
            image = attention_maps[:, :, i]
            image = 255 * image / image.max()
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.numpy().astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((256, 256)))
            image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
            images.append(image)

    num_images = len(images)
    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))

    # 创建一个空白的大图像
    total_width = cols * 256
    total_height = rows * 256
    combined_image = Image.new('RGB', (total_width, total_height))

    # 将每个图像粘贴到大图像中
    for i, image in enumerate(images):
        x_offset = (i % cols) * 256
        y_offset = (i // cols) * 256
        combined_image.paste(Image.fromarray(image), (x_offset, y_offset))
    # 保存大图像
    combined_image.save(filename)
    
