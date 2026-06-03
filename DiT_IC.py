import torch 
import torch.nn as nn

from ELIC.elic_official import ELIC
from LatentCodec import latent_codec
from diffusers import AutoencoderDC, SanaTransformer2DModel


class DiT_IC(nn.Module):
    def __init__(self, elic_path):
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


        

    def compress(self, img):
        latent_1 = self.vae.encode(img).latent * self.vae.config.scaling_factor
        latent_2 = self.e_aux((img + 1) / 2)

        compress_dict = self.latent_codec.compress(latent_1, latent_2) 

        return compress_dict # strings(列表中的列表)/z_shape："strings": [y_strings, z_strings],"z_shape": z_shape


if __name__ == '__main__':
    elic_path = "/img_research/StableCodec/ELIC/elic_official.pth"
