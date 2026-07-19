import torch 
import torch.nn as nn

from ELIC.elic_official import ELIC
from LatentCodec import latent_codec
from diffusers import AutoencoderDC, SanaTransformer2DModel
from scheduler import make_1step_sched, scheduler, trans_variance, randn_tensor


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
        self.e_aux = elic.g_a
        self.elic.eval()
        self.e_aux.requires_grad_(False)

        # latent codec
        self.latent_codec = latent_codec()
        

        print("------------------load denoised module-------------------")
        self.DiT = SanaTransformer2DModel.from_pretrained(dit_path, subfolder="transformer")
        self.codec_mode = codec_mode

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
        latent_2 = self.e_aux((img + 1) / 2)

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
        
        





        
        



if __name__ == '__main__':
    elic_path = "/img_research/StableCodec/ELIC/elic_official.pth"
