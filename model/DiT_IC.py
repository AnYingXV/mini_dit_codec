import torch 
import torch.nn as nn
import torch.nn.functional as F

from ELIC.elic_official import ELIC
from LatentCodec import latent_codec
from peft import get_peft_model, LoraConfig
from diffusers import AutoencoderDC, SanaTransformer2DModel
from scheduler import make_1step_sched, Scheduler, randn_tensor

# 筛VAE里decoder里需要插入LoRA的层
def filter_supported_modules(model):
    import re, torch.nn as nn
    pattern = re.compile(r"^decoder\..*(conv1|conv2|conv_in|conv_shortcut|conv_inverted|conv_point|to_k|to_q|to_v|to_out\.0)$")
    supported = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)
    return [n for n, m in model.named_modules() if pattern.match(n) and isinstance(m, supported)]

class LatentConditionAlignment(nn.Module):
    def __init__(self, in_channels = 320, SANADiT_emb_dim = 2304, num_tokens = 77, num_fixed = 48, CLIP_emb_dim = 768, transformer_depth: int = 2, transformer_heads: int = 8, if_train = True):
        super(LatentConditionAlignment, self).__init__()
        # in_channels应该是320
        self.SANADiT_emb_dim = SANADiT_emb_dim
        self.CLIP_emb_dim = CLIP_emb_dim
        self.num_tokens = num_tokens
        self.num_fixed = num_fixed
        self.if_train = if_train

        self.learnable_part = nn.Parameter(torch.randn(1, self.num_tokens - self.num_fixed, in_channels))
        self.align = nn.Sequential(
            nn.Linear(in_channels, SANADiT_emb_dim),
            nn.SiLU(),
            nn.Linear(SANADiT_emb_dim, SANADiT_emb_dim),
            nn.SiLU(),
            nn.Linear(SANADiT_emb_dim, SANADiT_emb_dim)
        )

        if if_train:
            self.align_clip = nn.Linear(SANADiT_emb_dim, CLIP_emb_dim) #对齐到CLIP的维度
            self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07))) # 论文公式里的τ

    def forward(self, latent, text_emb = None, img_emb = None):
        clip_align_loss = None
        B, C, H, W = latent.shape

        # 处理latent，使得和embedding形式对齐。图片：[B,C,W,H] embedding：[B,num_tokens,C]
        x = latent.flatten(2)
        x = F.adaptive_avg_pool1d(x, self.num_fixed)
        x = x.transpose(1, 2)
        learnable_part = self.learnable_part.expand(B, -1, -1)
        x = torch.cat([x, learnable_part], dim = 1)
        latent_prompt = self.align(x) # 维度从latent对齐到DiT，linear默认作用在最后一维
        # latent_prompt：[B, 77, 2304]
        # 如果不训练，就直接返回latent_prompt
        # 如果训练，则如下
        if self.if_train and (text_emb is not None or img_emb is not None):
            latent_mean = latent_prompt.mean(dim = 1) # [B, 2304]
            align_clip_latent = self.align_clip(latent_mean) # 将latent对齐到CLIP的维度
            align_clip_latent = F.normalize(align_clip_latent, dim = -1)

            # 训练对齐目标——CLIP文本embedding
            clip_target = text_emb if text_emb is not None else img_emb
            clip_target = F.normalize(clip_target.detach(), dim=-1)
            logit_scale = self.logit_scale.exp()

            latent_to_clip = logit_scale*align_clip_latent@clip_target.t()
            clip_to_latent = latent_to_clip.t()
            labels = torch.arange(B, device=latent.device)

            clip_align_loss = (F.cross_entropy(latent_to_clip, labels) + F.cross_entropy(clip_to_latent, labels)) / 2

        return (latent_prompt, clip_align_loss) if self.if_train else latent_prompt


class DiT_IC(nn.Module):
    def __init__(self, dit_path, elic_path):
        super(DiT_IC, self).__init__()
        print("------------------load encoders/decoder------------------")
        # vae
        self.vae = AutoencoderDC.from_pretrained(dit_path, subfolder="vae")

        # 冻结VAE
        for param in self.vae.parameters():
            param.requires_grad = False

        self.build_vae_lora(lora_rank=16, lora_alpha=16)
        
        # ELIC.ga
        elic = ELIC()
        checkpoint = torch.load(elic_path)
        elic.load_state_dict(checkpoint)
        elic.eval()
        self.E_aux = elic.g_a
        self.E_aux.requires_grad_(False)

        # latent codec
        self.latent_codec = latent_codec()
        

        print("------------------load denoised module-------------------")
        # 创建DiT预测、DiT LoRA、base scheduler、prompter
        self.time = 999
        self.build_DiT(dit_path=dit_path, device="cuda", use_merge=False)
        self.build_DiT_lora(lora_rank=16, lora_alpha=16)
        self.prompter = LatentConditionAlignment()
        
    def build_vae_lora(self, lora_rank, lora_alpha):
        vae_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=filter_supported_modules(self.vae),
            bias="none",
            init_lora_weights="gaussian",
        )
        self.vae = get_peft_model(self.vae, vae_lora_config)
        
        for param in self.vae.parameters():
            param.requires_grad = False
        
        for name, param in self.vae.named_parameters():
            if "lora" in name:
                param.requires_grad = True
        
        print("VAE-LoRA Done")

    def build_DiT(self, dit_path, device='cuda', use_merge=False):
        base_scheduler = make_1step_sched(dit_path, device)
        self.sched = Scheduler(base_scheduler)

        if use_merge:
            dit_config = SanaTransformer2DModel.load_config(dit_path, subfolder="transformer")
            self.DiT = SanaTransformer2DModel.from_config(dit_config)
        else:
            self.DiT = SanaTransformer2DModel.from_pretrained(dit_path, subfolder="transformer")
            # self.DiT = SanaTransformer2DModel.from_pretrained(dit_path, subfolder="transformer_512")

        self.register_buffer("timesteps", torch.tensor([self.time], dtype=torch.long))
        
        # 冻结DiT主干
        for name, param in self.DiT.named_parameters():
            param.requires_grad = False  
        
    def build_DiT_lora(self, lora_rank, lora_alpha, channel):
        target_modules_DiT = [
            "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
            "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj", "conv_inverted", "conv_point"
        ]
        DiT_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules_DiT,
            bias="none",
            init_lora_weights="gaussian",
        )
        self.DiT = get_peft_model(self.DiT, DiT_lora_config)
        
        # 先把所有module全部关掉
        for param in self.DiT.parameters():
            param.requires_grad = False
        
        # 只打开LoRA
        for name, param in self.DiT.named_parameters():
            if "lora" in name:
                param.requires_grad = True
                
        print("DiT-LoRA Done")

    def forward(self, img, text_emb, img_emb):
        latent_1 = self.vae.encode(img).latent * self.vae.config.scaling_factor
        latent_2 = self.E_aux((img + 1) / 2).detach()
        trans_log_variance, trans_y, y_aux, prompt, y_likelihoods, z_likelihoods = self.latent_codec(latent_1, latent_2)

        latent_hat = trans_y # 该模式对应论文自蒸馏创新点

        t = self.sched.base_scheduler.timesteps 
        expand_t = t.expand(latent_hat.shape[0]) 
        expand_t = expand_t * self.DiT.config.timestep_scale

        latent_prompt, clip_align_loss = self.prompter(prompt, text_emb, img_emb) # 训练模式下，返回tuple，接收返回的时候直接解包

        velocity = self.DiT(
            latent_hat, 
            encoder_hidden_states = latent_prompt.to(latent_hat.device), 
            timestep = expand_t.to(latent_hat.device),
            return_dict=False,
            )[0]

        x_denoised = self.sched.step(trans_log_variance, velocity, latent_hat) + y_aux

        output_image = self.vae.decode(x_denoised / self.vae.config.scaling_factor, return_dict=False)[0].clamp(-1, 1)

        # 0.05是该损失的权重，F.relu复制截断，m = 0.5，所以相似度达到0.5就停止对齐,loss一般是标量
        distill_loss = 0.05*F.relu(1 - 0.5 - F.cosine_similarity(x_denoised, latent_1.detach())).mean()

        return output_image, clip_align_loss, distill_loss, y_likelihoods, z_likelihoods


    def compress(self, img):
        latent_1 = self.vae.encode(img).latent * self.vae.config.scaling_factor
        latent_2 = self.E_aux((img + 1) / 2).detach()

        compress_dict = self.latent_codec.compress(latent_1, latent_2) 

        return compress_dict # strings(列表中的列表)/z_shape："strings": [y_strings, z_strings],"z_shape": z_shape

    # 预测速度场用的是SANA，实现一步去噪用的是DPMSolverMultistepScheduler
    def decompress(self, strings, z_shape):
        trans_log_variance, trans_y, y_aux, prompt = self.latent_codec.decompress(strings, z_shape)

        # latent 使用直接相等的模式，对应于自蒸馏创新点
        latent_hat = trans_y

        # 时间步
        t = self.sched.base_scheduler.timesteps # 获取生成的base_scheduler的时间步数值
        expand_t = t.expand(latent_hat.shape[0]) # batch维扩展，使得一个batch内的所有图片都能对应上时间步
        expand_t = expand_t * self.DiT.config.timestep_scale # 单步去噪模块和预测向量场模块使用的时间可能不在同一尺度上，但是二者之间有参数转换关系
        # latent 去噪条件
        latent_prompt, _ = self.prompter(prompt)

        # 预测向量场
        velocity = self.DiT(
            latent_hat, 
            encoder_hidden_states = latent_prompt.to(latent_hat.device), 
            timestep = expand_t.to(latent_hat.device),
            return_dict=False,
            )[0]

        # 一步去噪
        x_denoised = self.sched.step(trans_log_variance, velocity, latent_hat) + y_aux

        output_image = self.vae.decode(x_denoised / self.vae.config.scaling_factor, return_dict=False)[0].clamp(-1, 1)

        return output_image

        



if __name__ == '__main__':
    elic_path = "/img_research/StableCodec/ELIC/elic_official.pth"
