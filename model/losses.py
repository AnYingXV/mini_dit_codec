'''
每个loss值一定是标量, 不能是Tensor等
'''

import torch
import torch.nn as nn


class DiTICLosses(nn.Module):
    def __init__(self,
        lambda_mse,
        lambda_lpips,
        lambda_dists,
        lambda_distill,
        lambda_cond,
        lambda_adv,
    ):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_lpips = lambda_lpips
        self.lambda_dists = lambda_dists
        self.lambda_distill = lambda_distill
        self.lambda_cond = lambda_cond
        self.lambda_adv = lambda_adv

    def rate_loss(self, y_likelihoods, z_likelihoods, num_pixels):
        y_bpp = -torch.log2(y_likelihoods).sum() / num_pixels
        z_bpp = -torch.log2(z_likelihoods).sum() / num_pixels
        R = y_bpp + z_bpp
        return R

    def distortion_loss(self, mse_loss, lpips_loss, dists_loss):
        D = self.lambda_mse*mse_loss + self.lambda_lpips*lpips_loss + self.lambda_dists*dists_loss
        return D

    def alignment_loss(self, distill_loss, cond_loss):
        L_align = self.lambda_distill*distill_loss + self.lambda_cond*cond_loss
        return L_align

    def stage1(self,
        y_likelihoods,
        z_likelihoods,
        num_pixels,

        mse_loss,
        lpips_loss,
        dists_loss,

        distill_loss,
        cond_loss,
        lambda_rate,
    ):
        R = self.rate_loss(y_likelihoods, z_likelihoods, num_pixels)

        D = self.distortion_loss(mse_loss, lpips_loss, dists_loss)

        alignment = self.alignment_loss(distill_loss, cond_loss)

        total_loss1 = lambda_rate*R + D + alignment

        return total_loss1

    def stage2(self,
        y_likelihoods,
        z_likelihoods,
        num_pixels,

        mse_loss,
        lpips_loss,
        dists_loss,

        distill_loss,
        cond_loss,
        adversarial_loss,
        lambda_rate,
    ):
        losses1 = self.stage1(
            y_likelihoods=y_likelihoods,
            z_likelihoods=z_likelihoods,
            num_pixels=num_pixels,
            mse_loss=mse_loss,
            lpips_loss=lpips_loss,
            dists_loss=dists_loss,
            distill_loss=distill_loss,
            cond_loss=cond_loss,
            lambda_rate=lambda_rate,
        )

        total_loss2 = losses1 + self.lambda_adv*adversarial_loss

        return total_loss2


