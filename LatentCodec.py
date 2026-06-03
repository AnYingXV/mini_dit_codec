import torch 
import torch.nn as nn

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from utils_modules.modules import DepthConvBlock, ResidualBlockUpsample2, ResidualBlockWithStride2


# 创新点————时间步自适应预测
class SynthesisTransform2(nn.Module):
    def __init__(self, channel=320, channel_out=32) -> None:
        super().__init__()
        self.synthesis_transform = nn.Sequential(
            DepthConvBlock(channel, 320),
            DepthConvBlock(320, 320),
            DepthConvBlock(320, 320),
            Upsample(320, 192),
            nn.Conv2d(192, channel_out, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        x = self.synthesis_transform(x)
        return x

# 常规组件
class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.branch1 = self.down  # legacy name for loading old checkpoints

    def forward(self, x):
        return self.down(x)

class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=1, padding=0), 
            nn.PixelShuffle(2),
        )
        self.branch1 = self.up  # legacy name for loading old checkpoints

    def forward(self, x):
        return self.up(x)
# ga
class AnalysisTransform(nn.Module):
    def __init__(self, ch_emd=32, channel=320):
        super().__init__()
        self.pre1 = nn.Sequential(
            nn.Conv2d(ch_emd, 128, kernel_size=3, padding=1),
        )
            
        self.pre2 = nn.Sequential(
            Downsample(320, 64)
        )
        self.analysis_transform = nn.Sequential(
            DepthConvBlock(192, 192),
            DepthConvBlock(192, 192),
            Downsample(192, 320),
            DepthConvBlock(320, 320),
            DepthConvBlock(320, channel),
        )

    def forward(self, latent, latent2):
        x = torch.cat((self.pre1(latent), self.pre2(latent2)), dim=1)
        x = self.analysis_transform(x)
        return x
# gs
class SynthesisTransform(nn.Module):
    def __init__(self, channel=320, channel_out=32) -> None:
        super().__init__()
        self.synthesis_transform = nn.Sequential(
            DepthConvBlock(channel, 320),
            DepthConvBlock(320, 320),
            DepthConvBlock(320, 320),
            Upsample(320, 320),
            nn.Conv2d(320, channel_out, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        x = self.synthesis_transform(x)
        return x
# D_aux
class AuxDecoder(nn.Module):
    def __init__(self, ch_emd=32, channel=320) -> None:
        super().__init__()
        self.block = nn.Sequential(
            DepthConvBlock(channel, 320),
            DepthConvBlock(320, 320),
            Upsample(320, 320),
            nn.Conv2d(320, ch_emd, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.block(x)
        return x
# ha
class HyperAnalysis(nn.Module):
    def __init__(self, channel=320) -> None:
        super().__init__()
        self.reduction = nn.Sequential(
            DepthConvBlock(channel, channel),
            DepthConvBlock(channel, channel // 2),
            ResidualBlockWithStride2(channel // 2, channel // 2),
            ResidualBlockWithStride2(channel // 2, channel // 2),
        )

    def forward(self, x):
        x = self.reduction(x)
        return x
# hs
class HyperSynthesis(nn.Module):
    def __init__(self, channel=320) -> None:
        super().__init__()
        self.increase = nn.Sequential(
            ResidualBlockUpsample2(channel // 2, channel // 2),
            ResidualBlockUpsample2(channel // 2, channel // 2),
            DepthConvBlock(channel//2, channel),
            DepthConvBlock(channel, channel),
        )

    def forward(self, x):
        x = self.increase(x)
        return x

class CheckboardMaskedConv2d(nn.Conv2d):
    """
    if kernel_size == (5, 5)
    then mask:
        [[0., 1., 0., 1., 0.],
        [1., 0., 1., 0., 1.],
        [0., 1., 0., 1., 0.],
        [1., 0., 1., 0., 1.],
        [0., 1., 0., 1., 0.]]
    0: non-anchor
    1: anchor
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer("mask", torch.zeros_like(self.weight.data))

        self.mask[:, :, 0::2, 1::2] = 1
        self.mask[:, :, 1::2, 0::2] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        out = super().forward(x)

        return out  

class Adapter(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, (in_ch + out_ch) // 2, 3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d((in_ch + out_ch) // 2, (in_ch + out_ch) // 2, 3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d((in_ch + out_ch) // 2, out_ch, 3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.branch1(x)

class SpatialContext(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.block = nn.Sequential(
            DepthConvBlock(in_ch, in_ch),
            DepthConvBlock(in_ch, in_ch),
            DepthConvBlock(in_ch, in_ch),
            nn.Conv2d(in_ch, in_ch, 1),
        )

    def forward(self, x):
        context = self.block(x)
        return context

class LRP(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.block = nn.Sequential(
            Adapter(in_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)

class latent_codec(nn.Module):
    def __init__(self,):
        super(latent_codec, self).__init__()
        self.ga = AnalysisTransform()
        self.ha = HyperAnalysis()
        self.entropybottleneck = EntropyBottleneck()

    def get_mask(self, b, c, h, w, device="cuda"):
        patch0 = torch.tensor(((1., 0.), (0., 0.)), device = device) # 加上"."，用浮点型，而不是整型
        mask0 = patch0.repeat((h+1)//2, (w+1)//2)
        mask0 = mask0[:h, :w]
        mask0 = mask0.unsqueeze(0).unsqueeze(0)
        mask0 = mask0.expand(b, c//4, -1, -1)

        patch1 = torch.tensor(((0., 1.), (0., 0.)), device = device) 
        mask1 = patch1.repeat((h+1)//2, (w+1)//2)
        mask1 = mask1[:h, :w]
        mask1 = mask1.unsqueeze(0).unsqueeze(0)
        mask1 = mask1.expand(b, c//4, -1, -1)

        patch2 = torch.tensor(((0., 0.), (1., 0.)), device = device) 
        mask2 = patch2.repeat((h+1)//2, (w+1)//2)
        mask2 = mask2[:h, :w]
        mask2 = mask2.unsqueeze(0).unsqueeze(0)
        mask2 = mask2.expand(b, c//4, -1, -1)

        patch3 = torch.tensor(((0., 0.), (0., 1.)), device = device) 
        mask3 = patch3.repeat((h+1)//2, (w+1)//2)
        mask3 = mask3[:h, :w]
        mask3 = mask3.unsqueeze(0).unsqueeze(0)
        mask3 = mask3.expand(b, c//4, -1, -1)

        mask_0 = torch.cat((mask0, mask1, mask2, mask3), dim = 1)
        mask_1 = torch.cat((mask1, mask2, mask3, mask0), dim = 1)
        mask_2 = torch.cat((mask2, mask3, mask0, mask1), dim = 1)
        mask_3 = torch.cat((mask3, mask0, mask1, mask2), dim = 1)

        return mask_0, mask_1, mask_2, mask_3


    def compress(self, latent1, latent2):
        y = self.ga(latent1, latent2)
        z = self.ha(y)
        z_strings = self.entropybottleneck.compress(z)

        # 获取mask
        b, c, h, w = y.shape
        mask0, mask1, mask2, mask3 = get_mask(b, c, h, w)
