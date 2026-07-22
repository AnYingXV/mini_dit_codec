import torch 
import torch.nn as nn

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from utils_modules.modules import DepthConvBlock, ResidualBlockUpsample2, ResidualBlockWithStride2


# 常规组件
def ste_round(x):
    """Differentiable quantization via the Straight-Through-Estimator."""
    # STE (straight-through estimator) trick: x_hard - x_soft.detach() + x_soft
    return (torch.round(x) - x).detach() + x

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
# ga和gs的实质都是生成域和压缩域的转换
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
    def __init__(self, ch_emd=32, channel=320, channel_out=320):
        super(latent_codec, self).__init__()
        context_dim = channel * 2
        self.ga = AnalysisTransform(ch_emd, channel)
        self.gs = SynthesisTransform(channel, channel_out)
        self.gc = SpatialContext(in_ch=context_dim)
        self.ha = HyperAnalysis(channel)
        self.hs = HyperSynthesis(channel)
        self.entropybottleneck = EntropyBottleneck(channel//2)
        self.gaussianconditional = GaussianConditional(None)
        self.adapters_in = nn.ModuleList(Adapter(in_ch=channel, out_ch=context_dim) for i in range(4))
        self.adapters_out = nn.ModuleList(Adapter(in_ch=context_dim, out_ch=channel * 2) for i in range(4))
        self.LRP = nn.ModuleList(LRP(in_ch=context_dim, out_ch=channel) for i in range(4))
        self.D_aux = AuxDecoder(ch_emd, channel)
        self.prompt = SynthesisTransform(channel, 320)

    # 获取掩码√
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

    # 压缩通道√
    def squeeze_with_mask(self, sth, mask):
        sth0, sth1, sth2, sth3 = sth.chunk(4, 1)
        mask0, mask1, mask2, mask3 = mask.chunk(4, 1)
        squeezed = sth0*mask0 + sth1*mask1 + sth2*mask2 + sth3*mask3
        return squeezed
    
    # 恢复通道√
    def unsqueeze_with_mask(self, squeezed_sth, mask):
        mask0, mask1, mask2, mask3 = mask.chunk(4, 1)
        unsqueezed = torch.cat((squeezed_sth*mask0, squeezed_sth*mask1, squeezed_sth*mask2, squeezed_sth*mask3), dim = 1)
        return unsqueezed

    def one_step_forward(latent, means, scales, mask):
        squeeze_means = self.squeeze_with_mask(means, mask)
        squeeze_scales = self.squeeze_with_mask(scales, mask)
        squeeze_latent = self.squeeze_with_mask(latent, mask)

        _, y_likelihoods = self.gaussianconditional(squeeze_latent, squeeze_scales, means=squeeze_means)
        y_hat = ste_round(squeeze_latent - squeeze_means) + squeeze_means

        unsq_y_hat = unsqueeze_with_mask(y_hat, mask)
        unsq_y_likelihoods = unsqueeze_with_mask(y_likelihoods, mask)
        return unsq_y_hat, unsq_y_likelihoods

    # 得到待编码的量化残差、用于编码的cdf索引、返回用于下一步预测的y_hat
    # 此时传入的latent, means, scales都是完整的，而非经过对应mask处理后的稀疏的
    def prepare_encode(self, entropy_model, latent, means, scales, mask, symbol_list): 
        squeeze_means = self.squeeze_with_mask(means, mask)
        squeeze_scales = self.squeeze_with_mask(scales, mask)
        squeeze_latent = self.squeeze_with_mask(latent, mask)

        quantized_res = entropy_model.quantize(squeeze_latent, "symbols", squeeze_means)
        symbol_list.extend(quantized_res.reshape(-1).tolist())

        y_hat = self.unsqueeze_with_mask(squeeze_means + quantized_res, mask)
        return y_hat # 得到的是单步对应的y_hat

    def one_step_decode(self, entropy_model, means, scales, decoder, cdf, cdf_lengths, offsets, mask):
        squeeze_means = self.squeeze_with_mask(means, mask)
        squeeze_scales = self.squeeze_with_mask(scales, mask)

        # 解码出量化残差
        cdf_indexs = self.gaussianconditional.build_indexes(squeeze_scales)
        quantized_res = decoder.decode_stream(cdf_indexes.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
        quantized_res = torch.Tensor(quantized_res).reshape(squeeze_scales.shape).to(scales.device)

        y_hat = unsqueeze_with_mask(squeeze_means + quantized_res, mask)

        return y_hat

    def forward(self, latent1, latent2):
        y = self.ga(latent1, latent2)
        z = self.ha(y)

        _, z_likelihoods = self.entropybottleneck(z)
        z_offset = self.entropybottleneck._get_medians()
        z_hat = ste_round(z - z_offset) + z_offset

        b, c, w, h = y.shape
        mask0, mask1, mask2, mask3 = self.get_mask(b, c, h, w, device=y.device)

        base = self.hs(z_hat)
        mean_0, scale_0 = self.adapters_out[0](self.gc(self.adapters_in[0](base))).chunk(2, 1)
        y_hat_0, y_likelihoods_0 = self.one_step_forward(y, mean_0, scale_0, mask0)
        LRP = self.LRP[0](torch.cat((y_hat_0, base), dim = 1))*mask0
        LRP = 0.5 * torch.tanh(LRP)
        y_hat_0 = y_hat_0 + LRP

        base = base * (1 - mask0) + y_hat_0
        mean_1, scale_1 = self.adapters_out[1](self.gc(self.adapters_in[1](base))).chunk(2, 1)
        y_hat_1, y_likelihoods_1 = self.one_step_forward(y, mean_1, scale_1, mask1)
        LRP = self.LRP[1](torch.cat((y_hat_1, base), dim = 1))*mask1
        LRP = 0.5 * torch.tanh(LRP)
        y_hat_1 = y_hat_1 + LRP

        base = base * (1 - mask1) + y_hat_1
        mean_2, scale_2 = self.adapters_out[2](self.gc(self.adapters_in[2](base))).chunk(2, 1)
        y_hat_2, y_likelihoods_2 = self.one_step_forward(y, mean_2, scale_2, mask2)
        LRP = self.LRP[2](torch.cat((y_hat_2, base), dim = 1))*mask2
        LRP = 0.5 * torch.tanh(LRP)
        y_hat_2 = y_hat_2 + LRP

        base = base * (1 - mask2) + y_hat_2
        mean_3, scale_3 = self.adapters_out[3](self.gc(self.adapters_in[3](base))).chunk(2, 1)
        y_hat_3, y_likelihoods_3 = self.one_step_forward(y, mean_3, scale_3, mask3)
        LRP = self.LRP[3](torch.cat((y_hat_3, base), dim = 1))*mask3
        LRP = 0.5 * torch.tanh(LRP)
        y_hat_3 = y_hat_3 + LRP

        y_hat = y_hat_0 + y_hat_1 + y_hat_2 + y_hat_3
        y_likelihoods = y_likelihoods_0 + y_likelihoods_1 + y_likelihoods_2 + y_likelihoods_3
        scale_all = scale_0*mask0 + scale_1*mask1 + scale_2*mask2 + scale_3*mask3
        mean = self.gs(y_hat)
        y_aux = self.D_aux(y_hat)
        prompt = self.prompt(y_hat)

        return scale_all, mean, y_aux, prompt, y_likelihoods, z_likelihoods


    def compress(self, latent1, latent2):
        y = self.ga(latent1, latent2)
        z = self.ha(y)
        z_strings = self.entropybottleneck.compress(z)
        z_hat = self.entropybottleneck.decompress(z_strings, z.size()[-2:])

        # BufferedRansEncoder会保存状态，所以如果定义在init里可能导致每次调用时旧的状态没有被清除。保险起见在compress里定义，每次都重新创建一次
        y_encoder = BufferedRansEncoder() # 仅支持CDF索引和符号以list输入，而非Tensor

        cdf_indexs = []
        symbols = []
        y_strings = []


        # 获取mask
        b, c, h, w = y.shape
        mask0, mask1, mask2, mask3 = self.get_mask(b, c, h, w, device=y.device)

        # mean用来计算量化残差，scale用来选CDF表
        base = self.hs(z_hat)
        mean_0, scale_0 = self.adapters_out[0](self.gc(self.adapters_in[0](base))).chunk(2, 1)
        y_hat_0 = self.prepare_encode(self.gaussianconditional, y, mean_0, scale_0, mask0, symbols)
        LRP = self.LRP[0](torch.cat((y_hat_0, base), dim = 1))*mask0
        LRP = 0.5 * torch.tanh(LRP)
        y_hat_0 = y_hat_0 + LRP

        base = base * (1 - mask0) + y_hat_0
        mean_1, scale_1 = self.adapters_out[1](self.gc(self.adapters_in[1](base))).chunk(2, 1)
        y_hat_1 = self.prepare_encode(self.gaussianconditional, y, mean_1, scale_1, mask1, symbols)
        LRP = self.LRP[1](torch.cat((y_hat_1, base), dim = 1))*mask1
        LRP = 0.5 * torch.tanh(LRP)
        y_hat_1 = y_hat_1 + LRP

        base = base * (1 - mask1) + y_hat_1
        mean_2, scale_2 = self.adapters_out[2](self.gc(self.adapters_in[2](base))).chunk(2, 1)
        y_hat_2 = self.prepare_encode(self.gaussianconditional, y, mean_2, scale_2, mask2, symbols)
        LRP = self.LRP[2](torch.cat((y_hat_2, base), dim = 1))*mask2
        LRP = 0.5 * torch.tanh(LRP)
        y_hat_2 = y_hat_2 + LRP

        base = base * (1 - mask2) + y_hat_2
        mean_3, scale_3 = self.adapters_out[3](self.gc(self.adapters_in[3](base))).chunk(2, 1)
        y_hat_3 = self.prepare_encode(self.gaussianconditional, y, mean_3, scale_3, mask3, symbols)
        LRP = self.LRP[3](torch.cat((y_hat_3, base), dim = 1))*mask3
        LRP = 0.5 * torch.tanh(LRP)
        y_hat_3 = y_hat_3 + LRP

        scale_all = scale_0*mask0 + scale_1*mask1 + scale_2*mask2 + scale_3*mask3
        cdf_indexs = self.gaussianconditional.build_indexes(scale_all)
        cdf_list.extend(cdf_indexs.reshape(-1).tolist())
        cdf = self.gaussianconditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussianconditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussianconditional.offset.reshape(-1).int().tolist()


        # y的编码器是BufferedRansEncoder()，即self.y_encoder = BufferedRansEncoder()，encode_with_indexes是这个编码器的一个方法
        # self.y_encoder.encode_with_indexes组合起来就是说用这个编码器的encode_with_indexes方法对y进行编码
        y_encoder.encode_with_indexes(symbols, cdf_indexs, cdf, cdf_lengths, offsets)
        y_string = y_encoder.flush()
        y_strings.append(y_string)

        return {
            "strings" : [y_strings, z_strings],
            "shape" : z.size()[-2:],
        }
        

    def decompress(self, strings, z_shape):
        y_strings = strings[0][0]
        z_strings = strings[1]
        z_hat = self.entropybottleneck.decompress(z_strings, z_shape)
        cdf_indexs = []

        y_decoder = RansDecoder()
        cdf = self.gaussianconditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussianconditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussianconditional.offset.reshape(-1).int().tolist()

        b, c, h, w = z_hat.shape
        mask0, mask1, mask2, mask3 = self.get_mask(b, c * 2, h * 4, w * 4, device=z_hat.device)

        base = self.hs(z_hat)
        mean_0, scale_0 = self.adapters_out[0](self.gc(self.adapters_in[0](base))).chunk(2, 1)
        y_hat_0 = one_step_decode(self.gaussianconditional, mean_0, scale_0, y_decoder, cdf, cdf_lengths, offsets, mask0)
        LRP = self.LRP[0](torch.cat((y_hat_0, base), dim = 1))*mask0
        LRP = 0.5 * torch.tanh(LRP)
        y_hat_0 = y_hat_0 + LRP

        base = base * (1 - mask0) + y_hat_0
        mean_1, scale_1 = self.adapters_out[1](self.gc(self.adapters_in[1](base))).chunk(2, 1)
        y_hat_1 = one_step_decode(self.gaussianconditional, mean_1, scale_1, y_decoder, cdf, cdf_lengths, offsets, mask1)
        LRP = self.LRP[1](torch.cat((y_hat_1, base), dim = 1))*mask1
        LRP = 0.5 * torch.tanh(LRP)
        y_hat_1 = y_hat_1 + LRP

        base = base * (1 - mask1) + y_hat_1
        mean_2, scale_2 = self.adapters_out[2](self.gc(self.adapters_in[2](base))).chunk(2, 1)
        y_hat_2 = one_step_decode(self.gaussianconditional, mean_2, scale_2, y_decoder, cdf, cdf_lengths, offsets, mask2)
        LRP = self.LRP[2](torch.cat((y_hat_2, base), dim = 1))*mask2
        LRP = 0.5 * torch.tanh(LRP)
        y_hat_2 = y_hat_2 + LRP

        base = base * (1 - mask2) + y_hat_2
        mean_3, scale_3 = self.adapters_out[3](self.gc(self.adapters_in[3](base))).chunk(2, 1)
        y_hat_3 = one_step_decode(self.gaussianconditional, mean_3, scale_3, y_decoder, cdf, cdf_lengths, offsets, mask3)
        LRP = self.LRP[3](torch.cat((y_hat_3, base), dim = 1))*mask3
        LRP = 0.5 * torch.tanh(LRP)
        y_hat_3 = y_hat_3 + LRP

        y_hat = y_hat_0 + y_hat_1 + y_hat_2 + y_hat_3
        scale_all = scale_0*mask0 + scale_1*mask1 + scale_2*mask2 + scale_3*mask3 # 预测均值
        mean = self.gs(y_hat) # gs输出
        y_aux = self.D_aux(y_hat) # 辅助解码器输出
        prompt = self.prompt(y_hat) # 用于latent条件生成

        return scale_all, mean, y_aux, prompt