import torch 
import torch.nn as nn
import torch.nn.functional as F

from ELIC.elic_official import ELIC
from LatentCodec import latent_codec
from diffusers import AutoencoderDC, SanaTransformer2DModel
from scheduler import make_1step_sched, scheduler, trans_variance, randn_tensor

class LatentConditionAlignment(nn.Module):
    def __init__(self, SANADiT_emb_dim = 2304, num_tokens = 77, num_fixed = 48, CLIP_emb_dim = 768, in_channels, transformer_depth: int = 2, transformer_heads: int = 8, if_train = False):
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
        B, C, H, W = latent.shape

        # 处理latent，使得和embedding形式对齐。图片：[B,C,W,H] embedding：[B,num_tokens,C]
        x = latent.flatten(2)
        x = F.adaptive_avg_pool1d(x, self.num_fixed)
        x = transpose(1, 2)
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
            clip_target = F.normalize(clip_target, dim=-1)
            logit_scale = self.logit_scale.exp()

            latent_to_clip = logit_scale*align_clip_latent@clip_target.t()
            clip_to_latent = latent_to_clip.t()
            labels = torch.range(B, device=latent.device)

            clip_align_loss = (F.cross_entropy(latent_to_clip, labels) + F.cross_entropy(clip_to_latent, labels)) / 2

            return (latent_prompt, clip_align_loss) if self.if_train else latent_prompt


class DiT_IC(nn.Module):
    def __init__(self, dit_path, elic_path, codec_mode='self_dist'):
        super(DiT_IC, self).__init__()
        print("------------------load encoders/decoder------------------")
        # vae
        self.vae = AutoencoderDC.from_pretrained(dit_path, subfolder="vae")
        
        # ELIC.ga
        elic = ELIC()
        checkpoint = torch.load(elic_path)
        elic.load_state_dict(checkpoint)
        self.E_aux = elic.g_a
        self.elic.eval()
        self.E_aux.requires_grad_(False)

        # latent codec
        self.latent_codec = latent_codec()
        

        print("------------------load denoised module-------------------")
        self.DiT = SanaTransformer2DModel.from_pretrained(dit_path, subfolder="transformer")
        self.codec_mode = codec_mode
        self.prompter = LatentConditionAlignment()

    def build_DiT(self, dit_path, device='cuda', use_merge=False):
        base_scheduler = make_1step_sched(dit_path, device)
        self.sched = scheduler(base_scheduler, device)

        if use_merge:
            dit_config = SanaTransformer2DModel.load_config(dit_path, subfolder="transformer")
            self.DiT = SanaTransformer2DModel.from_config(dit_config)
        else:
            self.DiT = SanaTransformer2DModel.from_pretrained(dit_path, subfolder="transformer")
            # self.DiT = SanaTransformer2DModel.from_pretrained(dit_path, subfolder="transformer_512")

        self.register_buffer("timesteps", torch.tensor([self.time], dtype=torch.long))
        
        for name, param in self.DiT.named_parameters():
            param.requires_grad = False  
        
    def compress(self, img):
        latent_1 = self.vae.encode(img).latent * self.vae.config.scaling_factor
        latent_2 = self.E_aux((img + 1) / 2)

        compress_dict = self.latent_codec.compress(latent_1, latent_2) 

        return compress_dict # strings(列表中的列表)/z_shape："strings": [y_strings, z_strings],"z_shape": z_shape

    # 预测速度场用的是SANA，实现一步去噪用的是DPMSolverMultistepScheduler
    def decompress(self, strings, z_shape):
        log_variance, mean, y_aux, prompt = self.latent_codec(strings, z_shape)
        trans_log_variance = trans_variance(log_variance)
        scale = torch.exp(0.5 * trans_log_variance)

        # latent
        if self.codec_mode == 'sample':
            sample = randn_tensor(
                mean.shape, generator=None, device=mean.device, dtype=mean.dtype
            )
            latent_hat = mean + 0.1 * scale * sample
        else: 
            latent_hat = mean

        # 时间步
        t = self.sched.base_scheduler.timesteps # 获取生成的base_scheduler的时间步数值
        expand_t = t.expand(latent_hat.shape[0]) # batch维扩展，使得一个batch内的所有图片都能对应上时间步
        expand_t = expand_t * self.DiT.config.timestep_scale # 单步去噪模块和预测向量场模块使用的时间可能不在同一尺度上，但是二者之间有参数转换关系
        # latent 去噪条件
        prompt = self.prompter(prompt)

        # 预测向量场
        velocity = self.DiT(
            latent_hat, 
            encoder_hidden_states = prompt.to(latent_hat.device), 
            timestep = expand_t.to(latent_hat.device),
            return_dict=False,
            )[0]

        # 一步去噪
        x_denoised = self.sched.step(log_variance, velocity, latent_hat) + y_aux

        output_image = self.vae.decode(x_denoised / self.vae.config.scaling_factor, return_dict=False)[0].clamp(-1, 1)

        return output_image

        



if __name__ == '__main__':
    elic_path = "/img_research/StableCodec/ELIC/elic_official.pth"
