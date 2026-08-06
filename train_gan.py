"""
DiT-IC stage-2 adversarial DDP trainer adapted to the user's interfaces.

The official update order is preserved:
    1. update the codec / DiT generator
    2. update the entropy-bottleneck auxiliary parameters
    3. update EMA
    4. update the discriminator
    5. step generator and discriminator LR schedulers

Generator objective:
    stage1_rate_distortion_alignment_loss
    + adaptive_adversarial_weight * lambda_adv * generator_GAN_loss

Discriminator objective:
    hinge discriminator loss, enabled at loss.disc_start
    (for 6000->9000 stage-2 training, disc_start=6900 starts GAN after 30%)

This version is intentionally restricted to one GPU because the requested
hardware is one NVIDIA L40. Launch it through torchrun with one process.

Expected project layout:
    model/DiT_IC.py
    model/LatentCodec.py
    model/losses.py
    model/scheduler.py
    ELIC/elic_official.py
    utils/flickr8k_dataset_split.py

Launch:
    torchrun --standalone --nproc_per_node=1 \
      train_ditic_gan_ddp_flickr8k_split.py \
      --config train_ditic_gan_flickr8k_split_l40.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from utils.flickr8k_dataset_split import Flickr8kDataset
from model.DiT_IC import DiT_IC
from model.losses import DiTICLosses
from DISTS_pytorch import DISTS
from torch.utils.tensorboard import SummaryWriter


# ---------------------------------------------------------------------------
# Optional perceptual losses
# ---------------------------------------------------------------------------

class LPIPSLoss(nn.Module):
    """Scalar LPIPS loss. Dependency: pip install lpips."""

    def __init__(self, net: str = "alex") -> None:
        super().__init__()
        try:
            import lpips
        except ImportError as exc:
            raise ImportError(
                "lambda_lpips > 0, but package `lpips` is not installed. "
                "Install it with `pip install lpips`, or set lambda_lpips=0."
            ) from exc

        self.metric = lpips.LPIPS(net=net)
        self.metric.eval()
        self.metric.requires_grad_(False)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return self.metric(prediction, target).mean()


class DISTSLoss(nn.Module):
    """
    Scalar DISTS loss.

    Supports the common `DISTS_pytorch` package interface:
        from DISTS_pytorch import DISTS
    """

    def __init__(self) -> None:
        super().__init__()
        try:
            from DISTS_pytorch import DISTS
        except ImportError as exc:
            raise ImportError(
                "lambda_dists > 0, but package `DISTS_pytorch` is not installed. "
                "Install the DISTS implementation used by your project, or set "
                "lambda_dists=0."
            ) from exc

        self.metric = DISTS()
        self.metric.eval()
        self.metric.requires_grad_(False)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        # Most DISTS implementations expect [0, 1].
        prediction_01 = (prediction + 1.0) * 0.5
        target_01 = (target + 1.0) * 0.5
        return self.metric(prediction_01, target_01).mean()



class FrozenCLIPTextEncoder(nn.Module):
    """Frozen CLIP text tower used to turn Flickr8k captions into embeddings."""

    def __init__(
        self,
        model_name: str,
        output_type: str = "projected",
        max_length: Optional[int] = None,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoTokenizer, CLIPTextModelWithProjection
        except ImportError as exc:
            raise ImportError(
                "Text alignment requires `transformers`. Install it with "
                "`pip install transformers sentencepiece`."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_model = CLIPTextModelWithProjection.from_pretrained(model_name)
        self.text_model.eval()
        self.text_model.requires_grad_(False)

        output_type = output_type.lower()
        if output_type not in {"projected", "hidden"}:
            raise ValueError(
                "model.clip_text_output must be either `projected` or `hidden`."
            )
        self.output_type = output_type
        self.max_length = int(
            max_length
            if max_length is not None
            else getattr(self.tokenizer, "model_max_length", 77)
        )

    @torch.no_grad()
    def forward(self, captions: Sequence[str], device: torch.device) -> Tensor:
        if len(captions) == 0:
            raise ValueError("Cannot encode an empty caption batch.")

        tokens = self.tokenizer(
            list(captions),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = {
            name: value.to(device, non_blocking=True)
            for name, value in tokens.items()
        }
        outputs = self.text_model(**tokens)

        if self.output_type == "projected":
            embedding = outputs.text_embeds
        else:
            embedding = outputs.text_model_output.pooler_output

        return F.normalize(embedding.float(), dim=-1)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

@dataclass
class Batch:
    image: Tensor
    caption: Optional[List[str]] = None
    text_emb: Optional[Tensor] = None
    img_emb: Optional[Tensor] = None


class AverageMeter:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)

    def update(self, value: float, n: int) -> None:
        self.sum += float(value) * n
        self.count += n


class ResizeIfSmall:
    """Upscale only when an image is too small for the requested crop."""

    def __init__(self, min_size: Tuple[int, int]) -> None:
        self.min_h = int(min_size[0])
        self.min_w = int(min_size[1])

    def __call__(self, image):
        width, height = image.size
        if height >= self.min_h and width >= self.min_w:
            return image

        scale = max(self.min_h / height, self.min_w / width)
        new_h = max(self.min_h, int(round(height * scale)))
        new_w = max(self.min_w, int(round(width * scale)))
        return transforms.Resize((new_h, new_w))(image)


def unwrap_model(model: nn.Module) -> DiT_IC:
    return model.module if isinstance(model, DDP) else model


def parse_batch(raw_batch: Any, device: torch.device) -> Batch:
    """
    Supported dataset outputs:
      1. image_tensor
      2. (image_tensor, caption/text_embedding)
      3. (image_tensor, text_embedding, image_embedding)
      4. {"image": ..., "caption": ..., "text_emb": ..., "img_emb": ...}

    Flickr8k's default DataLoader collation turns the caption strings into
    ``list[str]``. They remain on CPU and are encoded by the frozen CLIP text
    tower immediately before the DiT-IC forward pass.
    """
    if isinstance(raw_batch, Tensor):
        return Batch(image=raw_batch.to(device, non_blocking=True))

    if isinstance(raw_batch, Mapping):
        image = raw_batch.get("image", raw_batch.get("img"))
        if image is None:
            raise KeyError("Batch dict must contain `image` or `img`.")

        caption_value = raw_batch.get("caption", raw_batch.get("captions"))
        captions: Optional[List[str]]
        if caption_value is None:
            captions = None
        elif isinstance(caption_value, str):
            captions = [caption_value]
        elif isinstance(caption_value, (tuple, list)) and all(
            isinstance(item, str) for item in caption_value
        ):
            captions = list(caption_value)
        else:
            raise TypeError(
                "Batch `caption` must be a string or a sequence of strings, "
                f"got {type(caption_value)!r}."
            )

        text_emb = raw_batch.get("text_emb")
        img_emb = raw_batch.get("img_emb")
        return Batch(
            image=image.to(device, non_blocking=True),
            caption=captions,
            text_emb=None
            if text_emb is None
            else text_emb.to(device, non_blocking=True),
            img_emb=None
            if img_emb is None
            else img_emb.to(device, non_blocking=True),
        )

    if isinstance(raw_batch, (tuple, list)):
        if len(raw_batch) == 0:
            raise ValueError("Empty batch.")

        image = raw_batch[0].to(device, non_blocking=True)
        captions = None
        text_emb = None
        img_emb = None

        if len(raw_batch) >= 2:
            second = raw_batch[1]
            if isinstance(second, Tensor):
                text_emb = second.to(device, non_blocking=True)
            elif isinstance(second, str):
                captions = [second]
            elif isinstance(second, (tuple, list)) and all(
                isinstance(item, str) for item in second
            ):
                captions = list(second)

        if len(raw_batch) >= 3 and isinstance(raw_batch[2], Tensor):
            img_emb = raw_batch[2].to(device, non_blocking=True)

        return Batch(
            image=image,
            caption=captions,
            text_emb=text_emb,
            img_emb=img_emb,
        )

    raise TypeError(f"Unsupported batch type: {type(raw_batch)!r}")


def attach_text_embeddings(
    batch: Batch,
    text_encoder: Optional[FrozenCLIPTextEncoder],
    device: torch.device,
) -> Batch:
    """Encode captions only when precomputed text embeddings are unavailable."""
    if batch.text_emb is not None:
        return batch
    if batch.caption is None:
        return batch
    if text_encoder is None:
        raise RuntimeError(
            "The dataset returned captions, but the CLIP text encoder was not "
            "created. Set loss.lambda_cond > 0 and configure model.clip_text_model."
        )

    batch.text_emb = text_encoder(batch.caption, device)
    return batch

def scalar_or_zero(value: Optional[Tensor], reference: Tensor) -> Tensor:
    if value is None:
        return reference.new_zeros(())
    return value.mean() if value.ndim != 0 else value


def entropy_aux_loss(model: DiT_IC) -> Tensor:
    """
    CompressAI EntropyBottleneck auxiliary quantile loss.

    The user's `latent_codec` has no wrapper named `aux_loss`, so the official
    training call `model.codec.aux_loss()` is adapted to the actual interface.
    """
    bottleneck = model.latent_codec.entropybottleneck
    if hasattr(bottleneck, "loss"):
        loss = bottleneck.loss()
        return loss.mean() if loss.ndim != 0 else loss
    raise AttributeError(
        "latent_codec.entropybottleneck has no `loss()` method. "
        "Check the installed CompressAI version."
    )


def get_aux_parameters(model: DiT_IC) -> Iterable[nn.Parameter]:
    """
    The auxiliary optimizer should update only EntropyBottleneck quantiles.
    """
    parameters = [
        parameter
        for name, parameter in model.latent_codec.entropybottleneck.named_parameters()
        if name.endswith("quantiles") and parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError(
            "No trainable EntropyBottleneck quantile parameters were found."
        )
    return parameters


def get_main_parameters(model: DiT_IC) -> Iterable[nn.Parameter]:
    """
    All trainable parameters except EntropyBottleneck quantiles.

    This follows the freezing decisions already made inside DiT_IC:
      - E_aux frozen
      - VAE backbone frozen, VAE LoRA trainable
      - DiT backbone frozen, DiT LoRA trainable
      - latent_codec trainable
      - latent-condition projector trainable
    """
    aux_ids = {id(parameter) for parameter in get_aux_parameters(model)}
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in aux_ids
    ]
    if not parameters:
        raise RuntimeError("No main trainable parameters were found.")
    return parameters


@torch.no_grad()
def make_ema(model: DiT_IC) -> Dict[str, Tensor]:
    return {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }


@torch.no_grad()
def update_ema(
    ema_state: Dict[str, Tensor],
    model: DiT_IC,
    decay: float,
) -> None:
    current_state = model.state_dict()
    for name, current in current_state.items():
        if name not in ema_state or ema_state[name].shape != current.shape:
            ema_state[name] = current.detach().clone()
            continue

        ema_tensor = ema_state[name]
        if torch.is_floating_point(ema_tensor):
            ema_tensor.mul_(decay).add_(current.detach(), alpha=1.0 - decay)
        else:
            ema_tensor.copy_(current.detach())


def reduce_mean(value: Tensor) -> Tensor:
    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= dist.get_world_size()
    return reduced


def create_logger(output_dir: Path, rank: int) -> logging.Logger:
    logger = logging.getLogger("ditic_train")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    if rank != 0:
        logger.addHandler(logging.NullHandler())
        return logger

    output_dir.mkdir(parents=True, exist_ok=True)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / "train.log")
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(file_handler)
    return logger


def set_seed(seed: int, rank: int) -> None:
    process_seed = seed + rank
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed)


def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("The YAML root must be a mapping.")
    return config


def setup_ddp() -> Tuple[int, int, int]:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def save_checkpoint(
    path: Path,
    model: DiT_IC,
    criterion: "AdversarialCriterion",
    ema_state: Dict[str, Tensor],
    optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    aux_optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    discriminator_scheduler: LambdaLR,
    train_steps: int,
    config: Dict[str, Any],
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "discriminator": criterion.discriminator.state_dict(),
        "ema": ema_state,
        "optimizer": optimizer.state_dict(),
        "discriminator_optimizer": discriminator_optimizer.state_dict(),
        "aux_optimizer": aux_optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "discriminator_scheduler": discriminator_scheduler.state_dict(),
        "train_steps": train_steps,
        "config": config,
    }
    torch.save(checkpoint, path)


def override_optimizer_and_scheduler_lr(
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    new_lr: float,
) -> None:
    """
    Override a resumed optimizer LR without letting the restored LambdaLR
    scheduler change it back on the next scheduler.step().

    This is especially useful when entering stage 2 from a stage-1 checkpoint:
    the optimizer state (Adam moments) is retained, while the stage-2 YAML can
    deliberately select a smaller generator learning rate.
    """
    new_lr = float(new_lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
        param_group["initial_lr"] = new_lr

    scheduler.base_lrs = [new_lr for _ in optimizer.param_groups]
    scheduler._last_lr = [new_lr for _ in optimizer.param_groups]


def load_checkpoint(
    path: Path,
    model: DiT_IC,
    criterion: "AdversarialCriterion",
    optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    aux_optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    discriminator_scheduler: LambdaLR,
    device: torch.device,
) -> Tuple[int, Dict[str, Tensor], bool]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    resumed_discriminator = "discriminator" in checkpoint
    if resumed_discriminator:
        criterion.discriminator.load_state_dict(
            checkpoint["discriminator"], strict=True
        )
        discriminator_optimizer.load_state_dict(
            checkpoint["discriminator_optimizer"]
        )
        discriminator_scheduler.load_state_dict(
            checkpoint["discriminator_scheduler"]
        )

    ema_state = {
        name: tensor.to(device)
        for name, tensor in checkpoint["ema"].items()
    }
    return int(checkpoint["train_steps"]), ema_state, resumed_discriminator


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

class TrainingCriterion(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        loss_cfg = config["loss"]

        self.lambda_rate = float(loss_cfg["lambda_rate"])
        self.lambda_cond = float(loss_cfg.get("lambda_cond", 0.0))

        self.combiner = DiTICLosses(
            lambda_mse=float(loss_cfg.get("lambda_mse", 1.0)),
            lambda_lpips=float(loss_cfg.get("lambda_lpips", 0.0)),
            lambda_dists=float(loss_cfg.get("lambda_dists", 0.0)),
            lambda_distill=float(loss_cfg.get("lambda_distill", 1.0)),
            lambda_cond=self.lambda_cond,
            lambda_adv=0.0,
        )

        self.lpips_metric: Optional[nn.Module] = None
        self.dists_metric: Optional[nn.Module] = None

        if self.combiner.lambda_lpips > 0:
            self.lpips_metric = LPIPSLoss(loss_cfg.get("lpips_net", "alex"))

        if self.combiner.lambda_dists > 0:
            self.dists_metric = DISTSLoss()

    def forward(
        self,
        target: Tensor,
        reconstruction: Tensor,
        clip_align_loss: Optional[Tensor],
        distill_loss: Tensor,
        y_likelihoods: Tensor,
        z_likelihoods: Tensor,
        include_condition: bool = True,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        mse = F.mse_loss(reconstruction, target)

        lpips_value = (
            reconstruction.new_zeros(())
            if self.lpips_metric is None
            else self.lpips_metric(reconstruction, target)
        )
        dists_value = (
            reconstruction.new_zeros(())
            if self.dists_metric is None
            else self.dists_metric(reconstruction, target)
        )

        cond = (
            scalar_or_zero(clip_align_loss, reconstruction)
            if include_condition
            else reconstruction.new_zeros(())
        )
        if include_condition and self.lambda_cond > 0 and clip_align_loss is None:
            raise RuntimeError(
                "lambda_cond > 0, but the dataset did not provide `text_emb` "
                "or `img_emb`. Return embeddings from the dataset, or set "
                "loss.lambda_cond=0."
            )

        distill = scalar_or_zero(distill_loss, reconstruction)
        num_pixels = target.shape[0] * target.shape[-2] * target.shape[-1]

        total = self.combiner.stage1(
            y_likelihoods=y_likelihoods,
            z_likelihoods=z_likelihoods,
            num_pixels=num_pixels,
            mse_loss=mse,
            lpips_loss=lpips_value,
            dists_loss=dists_value,
            distill_loss=distill,
            cond_loss=cond,
            lambda_rate=self.lambda_rate,
        )

        with torch.no_grad():
            rate = self.combiner.rate_loss(
                y_likelihoods, z_likelihoods, num_pixels
            )
            psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))

        logs = {
            "total": total.detach(),
            "rate": rate.detach(),
            "mse": mse.detach(),
            "psnr": psnr.detach(),
            "lpips": lpips_value.detach(),
            "dists": dists_value.detach(),
            "distill": distill.detach(),
            "cond": cond.detach(),
        }
        return total, logs



# ---------------------------------------------------------------------------
# Adversarial loss components
# ---------------------------------------------------------------------------

class ActNorm(nn.Module):
    """Small ActNorm layer compatible with a PatchGAN discriminator."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.register_buffer("initialized", torch.tensor(False))
        self.eps = float(eps)

    @torch.no_grad()
    def initialize(self, x: Tensor) -> None:
        flatten = x.permute(1, 0, 2, 3).contiguous().view(x.shape[1], -1)
        mean = flatten.mean(dim=1).view(1, -1, 1, 1)
        std = flatten.std(dim=1).view(1, -1, 1, 1).clamp_min(self.eps)
        self.loc.copy_(-mean)
        self.scale.copy_(1.0 / std)
        self.initialized.fill_(True)

    def forward(self, x: Tensor) -> Tensor:
        if self.training and not bool(self.initialized):
            self.initialize(x)
        return self.scale * (x + self.loc)


def _normalization_layer(channels: int, use_actnorm: bool) -> nn.Module:
    if use_actnorm:
        return ActNorm(channels)
    return nn.BatchNorm2d(channels)


class NLayerDiscriminator(nn.Module):
    """
    PatchGAN discriminator used by the stage-2 adversarial objective.

    The network returns a spatial grid of logits rather than one image-level
    scalar, matching the usual image-codec/VQGAN adversarial training setup.
    """

    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        use_actnorm: bool = False,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("loss.disc_num_layers must be >= 1.")

        kw = 4
        padw = 1
        layers: List[nn.Module] = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        for layer_index in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** layer_index, 8)
            channels = ndf * nf_mult
            layers.extend(
                [
                    nn.Conv2d(
                        ndf * nf_mult_prev,
                        channels,
                        kernel_size=kw,
                        stride=2,
                        padding=padw,
                        bias=not use_actnorm,
                    ),
                    _normalization_layer(channels, use_actnorm),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        channels = ndf * nf_mult
        layers.extend(
            [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    channels,
                    kernel_size=kw,
                    stride=1,
                    padding=padw,
                    bias=not use_actnorm,
                ),
                _normalization_layer(channels, use_actnorm),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(channels, 1, kernel_size=kw, stride=1, padding=padw),
            ]
        )
        self.main = nn.Sequential(*layers)
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, 1.0, 0.02)
            nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x)


def hinge_discriminator_loss(
    logits_real: Tensor,
    logits_fake: Tensor,
) -> Tensor:
    loss_real = F.relu(1.0 - logits_real).mean()
    loss_fake = F.relu(1.0 + logits_fake).mean()
    return 0.5 * (loss_real + loss_fake)


def vanilla_discriminator_loss(
    logits_real: Tensor,
    logits_fake: Tensor,
) -> Tensor:
    loss_real = F.softplus(-logits_real).mean()
    loss_fake = F.softplus(logits_fake).mean()
    return 0.5 * (loss_real + loss_fake)


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def get_generator_last_layer(model: DiT_IC) -> nn.Parameter:
    """
    Return the final trainable VAE-decoder parameter used for adaptive GAN
    weighting. The VAE backbone is frozen in this project, so a decoder LoRA
    parameter is preferred.
    """
    preferred: List[Tuple[str, nn.Parameter]] = []
    fallback: List[Tuple[str, nn.Parameter]] = []

    for name, parameter in model.vae.named_parameters():
        if not parameter.requires_grad:
            continue
        fallback.append((name, parameter))
        if "decoder" in name and "lora" in name.lower():
            preferred.append((name, parameter))

    candidates = preferred or fallback
    if not candidates:
        raise RuntimeError(
            "No trainable VAE decoder parameter was found for adaptive "
            "adversarial weighting."
        )
    return candidates[-1][1]


class AdversarialCriterion(TrainingCriterion):
    """
    Stage-1 criterion plus the official two-optimizer adversarial interface.

    The discriminator is owned by the criterion, while the user's DiT_IC model
    keeps its existing forward signature unchanged.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        loss_cfg = config["loss"]

        self.lambda_adv = float(loss_cfg.get("lambda_adv", 1.0))
        self.disc_start = int(loss_cfg.get("disc_start", 0))
        self.disc_factor = float(loss_cfg.get("disc_factor", 1.0))
        self.disc_weight = float(loss_cfg.get("disc_weight", 1.0))
        self.adaptive_adv_weight = bool(
            loss_cfg.get("adaptive_adv_weight", True)
        )
        self.adaptive_weight_max = float(
            loss_cfg.get("adaptive_weight_max", 1.0e4)
        )

        self.discriminator = NLayerDiscriminator(
            input_nc=int(loss_cfg.get("disc_in_channels", 3)),
            ndf=int(loss_cfg.get("disc_ndf", 64)),
            n_layers=int(loss_cfg.get("disc_num_layers", 3)),
            use_actnorm=bool(loss_cfg.get("disc_use_actnorm", False)),
        )

        disc_loss_name = str(
            loss_cfg.get("disc_loss", "hinge")
        ).lower()
        if disc_loss_name == "hinge":
            self.discriminator_loss = hinge_discriminator_loss
        elif disc_loss_name in {"vanilla", "softplus"}:
            self.discriminator_loss = vanilla_discriminator_loss
        else:
            raise ValueError(
                "loss.disc_loss must be `hinge` or `vanilla`."
            )

    def adversarial_enabled(self, global_step: int) -> bool:
        return (
            global_step >= self.disc_start
            and self.lambda_adv > 0.0
            and self.disc_factor > 0.0
        )

    def generator_loss(
        self,
        target: Tensor,
        reconstruction: Tensor,
        clip_align_loss: Optional[Tensor],
        distill_loss: Tensor,
        y_likelihoods: Tensor,
        z_likelihoods: Tensor,
        global_step: int,
        last_layer: nn.Parameter,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        base_total, logs = super().forward(
            target=target,
            reconstruction=reconstruction,
            clip_align_loss=clip_align_loss,
            distill_loss=distill_loss,
            y_likelihoods=y_likelihoods,
            z_likelihoods=z_likelihoods,
            include_condition=True,
        )

        zero = reconstruction.new_zeros(())
        if not self.adversarial_enabled(global_step):
            logs.update(
                {
                    "generator_adv": zero.detach(),
                    "adaptive_adv_weight": zero.detach(),
                    "disc_factor": zero.detach(),
                }
            )
            return base_total, logs

        logits_fake = self.discriminator(reconstruction.contiguous())
        generator_adv = -logits_fake.mean()

        adaptive_weight = reconstruction.new_tensor(self.disc_weight)
        if self.adaptive_adv_weight:
            rec_grads = torch.autograd.grad(
                base_total,
                last_layer,
                retain_graph=True,
                allow_unused=True,
            )[0]
            gan_grads = torch.autograd.grad(
                generator_adv,
                last_layer,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if rec_grads is not None and gan_grads is not None:
                adaptive_weight = (
                    torch.norm(rec_grads)
                    / (torch.norm(gan_grads) + 1.0e-4)
                ).clamp(0.0, self.adaptive_weight_max).detach()
                adaptive_weight = adaptive_weight * self.disc_weight

        adversarial_term = (
            adaptive_weight
            * self.disc_factor
            * self.lambda_adv
            * generator_adv
        )
        total = base_total + adversarial_term
        logs["total"] = total.detach()
        logs.update(
            {
                "generator_adv": generator_adv.detach(),
                "adaptive_adv_weight": adaptive_weight.detach(),
                "disc_factor": reconstruction.new_tensor(
                    self.disc_factor
                ).detach(),
            }
        )
        return total, logs

    def discriminator_step_loss(
        self,
        target: Tensor,
        reconstruction: Tensor,
        global_step: int,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        zero = reconstruction.new_zeros(())
        if not self.adversarial_enabled(global_step):
            return zero, {
                "discriminator": zero.detach(),
                "logits_real": zero.detach(),
                "logits_fake": zero.detach(),
            }

        logits_real = self.discriminator(target.detach().contiguous())
        logits_fake = self.discriminator(
            reconstruction.detach().contiguous()
        )
        discriminator = (
            self.disc_factor
            * self.discriminator_loss(logits_real, logits_fake)
        )
        logs = {
            "discriminator": discriminator.detach(),
            "logits_real": logits_real.detach().mean(),
            "logits_fake": logits_fake.detach().mean(),
        }
        return discriminator, logs


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_dataloader(
    config: Dict[str, Any],
    rank: int,
    world_size: int,
    training: bool,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    data_cfg = config["data"]
    patch_size = tuple(data_cfg["patch_size"])

    if training:
        transform = transforms.Compose(
            [
                ResizeIfSmall(patch_size),
                transforms.RandomCrop(patch_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )
        split_name = "train"
        return_caption = True
        drop_last = True
        split_file = data_cfg.get("train_split_file")
    else:
        transform = transforms.Compose(
            [
                ResizeIfSmall(patch_size),
                transforms.CenterCrop(patch_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )
        split_name = "valid"
        return_caption = False
        drop_last = False
        split_file = data_cfg.get("valid_split_file")

    dataset = Flickr8kDataset(
        image_root=data_cfg["image_root"],
        annotation_file=data_cfg["annotation_file"],
        split=split_name,
        transform=transform,
        caption_indices=data_cfg.get(
            "train_caption_indices", [0, 1, 2, 3, 4]
        ),
        return_caption=return_caption,
        split_file=split_file,
        validation_ratio=float(data_cfg.get("validation_ratio", 0.125)),
        split_seed=int(data_cfg.get("split_seed", 903)),
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=training,
        drop_last=drop_last,
    )

    global_batch_size = (
        int(config["train"]["global_batch_size"])
        if training
        else int(
            config["train"].get(
                "valid_global_batch_size",
                config["train"]["global_batch_size"],
            )
        )
    )
    if global_batch_size % world_size != 0:
        raise ValueError(
            f"Global batch size {global_batch_size} must be divisible by "
            f"world size {world_size}."
        )

    num_workers = int(data_cfg.get("num_workers", 4))
    loader = DataLoader(
        dataset,
        batch_size=global_batch_size // world_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn,
    )
    return loader, sampler


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: TrainingCriterion,
    device: torch.device,
    text_encoder: Optional[FrozenCLIPTextEncoder],
) -> Dict[str, float]:
    model.eval()
    meters: Dict[str, AverageMeter] = {}

    for raw_batch in loader:
        batch = parse_batch(raw_batch, device)
        # Validation intentionally has no captions. The model reconstructs from
        # the compressed latent condition alone, and condition loss is skipped.
        outputs = model(batch.image, None, None)
        (
            reconstruction,
            clip_align_loss,
            distill_loss,
            y_likelihoods,
            z_likelihoods,
        ) = outputs

        _, logs = criterion(
            target=batch.image,
            reconstruction=reconstruction,
            clip_align_loss=clip_align_loss,
            distill_loss=distill_loss,
            y_likelihoods=y_likelihoods,
            z_likelihoods=z_likelihoods,
            include_condition=False,
        )

        n = batch.image.shape[0]
        for name, value in logs.items():
            meters.setdefault(name, AverageMeter()).update(value.item(), n)

    result: Dict[str, float] = {}
    for name, meter in meters.items():
        local_sum = torch.tensor(meter.sum, device=device, dtype=torch.float64)
        local_count = torch.tensor(meter.count, device=device, dtype=torch.float64)
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
        result[name] = (local_sum / local_count.clamp_min(1)).item()

    model.train()
    return result


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    rank: int,
    world_size: int,
    local_rank: int,
    config: Dict[str, Any],
) -> None:
    device = torch.device("cuda", local_rank)
    train_cfg = config["train"]
    optim_cfg = config["optimizer"]

    if world_size != 1:
        raise RuntimeError(
            "This L40 adaptation is configured for one GPU. Launch with "
            "`torchrun --standalone --nproc_per_node=1 ...`."
        )

    # ------------------------------------------------------------------
    # Output directories and TensorBoard
    # ------------------------------------------------------------------
    output_dir = Path(train_cfg["output_dir"]) / train_cfg["exp_name"]
    checkpoint_dir = output_dir / "checkpoints"
    tensorboard_dir = output_dir / "tensorboard"

    writer = None

    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        tensorboard_dir.mkdir(parents=True, exist_ok=True)

        writer = SummaryWriter(
            log_dir=str(tensorboard_dir)
        )

    dist.barrier()

    logger = create_logger(output_dir, rank)

    if rank == 0:
        logger.info(
            "Configuration:\n%s",
            json.dumps(config, indent=2),
        )
        logger.info(
            "TensorBoard directory: %s",
            tensorboard_dir,
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = DiT_IC(
        dit_path=config["model"]["dit_path"],
        elic_path=config["model"]["elic_path"],
    ).to(device)

    criterion = AdversarialCriterion(config).to(device)

    # ------------------------------------------------------------------
    # Frozen CLIP text encoder
    # ------------------------------------------------------------------
    text_encoder: Optional[FrozenCLIPTextEncoder] = None

    if criterion.lambda_cond > 0:
        model_cfg = config["model"]

        text_encoder = FrozenCLIPTextEncoder(
            model_name=model_cfg.get(
                "clip_text_model",
                "openai/clip-vit-base-patch32",
            ),
            output_type=model_cfg.get(
                "clip_text_output",
                "projected",
            ),
            max_length=model_cfg.get(
                "clip_max_length",
            ),
        ).to(device)

        text_encoder.eval()

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------
    main_parameters = list(get_main_parameters(model))
    aux_parameters = list(get_aux_parameters(model))

    optimizer = AdamW(
        main_parameters,
        lr=float(optim_cfg["lr"]),
        betas=(
            0.9,
            float(optim_cfg.get("beta2", 0.999)),
        ),
        weight_decay=float(
            optim_cfg.get("weight_decay", 0.0)
        ),
    )

    aux_optimizer = AdamW(
        aux_parameters,
        lr=float(optim_cfg["aux_lr"]),
        weight_decay=0.0,
    )

    discriminator_optimizer = AdamW(
        criterion.discriminator.parameters(),
        lr=float(optim_cfg.get("disc_lr", optim_cfg["lr"])),
        betas=tuple(
            float(value)
            for value in optim_cfg.get("beta_vae", [0.5, 0.9])
        ),
        weight_decay=float(optim_cfg.get("disc_weight_decay", 0.0)),
    )

    # ------------------------------------------------------------------
    # LR scheduler
    # ------------------------------------------------------------------
    milestones = list(
        map(int, optim_cfg.get("step_lr", []))
    )

    gammas = list(
        map(
            float,
            optim_cfg.get(
                "step_gamma",
                [0.5] * len(milestones),
            ),
        )
    )

    if len(gammas) != len(milestones):
        raise ValueError(
            "optimizer.step_gamma and optimizer.step_lr "
            "must have equal length."
        )

    def lr_lambda(step: int) -> float:
        factor = 1.0

        for milestone, gamma in zip(
            milestones,
            gammas,
        ):
            if step >= milestone:
                factor *= gamma

        return factor

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )

    discriminator_scheduler = LambdaLR(
        discriminator_optimizer,
        lr_lambda=lr_lambda,
    )

    # ------------------------------------------------------------------
    # EMA and resume
    # ------------------------------------------------------------------
    ema_state = make_ema(model)
    train_steps = 0

    resume_path = train_cfg.get("resume")

    if resume_path:
        resume_file = Path(resume_path)

        (
            train_steps,
            ema_state,
            resumed_discriminator,
        ) = load_checkpoint(
            resume_file,
            model,
            criterion,
            optimizer,
            discriminator_optimizer,
            aux_optimizer,
            scheduler,
            discriminator_scheduler,
            device,
        )

        # A stage-1 checkpoint contains optimizer/scheduler state with its old
        # learning rate. By default, when entering stage 2 from a no-GAN
        # checkpoint, keep the Adam moments but use optimizer.lr from this YAML.
        # For a later stage-2-to-stage-2 resume, the checkpoint LR is preserved
        # unless optimizer.override_resume_lr is explicitly set to true.
        override_resume_lr = bool(
            optim_cfg.get("override_resume_lr", not resumed_discriminator)
        )
        if override_resume_lr:
            override_optimizer_and_scheduler_lr(
                optimizer,
                scheduler,
                float(optim_cfg["lr"]),
            )

        # A resumed GAN checkpoint may also override the discriminator LR, but
        # this is opt-in so ordinary interruption recovery stays exact.
        override_resume_disc_lr = bool(
            optim_cfg.get("override_resume_disc_lr", False)
        )
        if resumed_discriminator and override_resume_disc_lr:
            override_optimizer_and_scheduler_lr(
                discriminator_optimizer,
                discriminator_scheduler,
                float(optim_cfg.get("disc_lr", optim_cfg["lr"])),
            )

        if rank == 0:
            logger.info(
                "Resumed from %s at step %d",
                resume_file,
                train_steps,
            )
            if not resumed_discriminator:
                logger.info(
                    "The checkpoint is a stage-1/no-GAN checkpoint. "
                    "Generator, main optimizer, auxiliary optimizer, LR "
                    "scheduler, EMA, and train_steps were restored; the "
                    "discriminator starts from a fresh initialization."
                )
            if override_resume_lr:
                logger.info(
                    "Generator LR overridden from the stage-2 YAML: %.6e",
                    optimizer.param_groups[0]["lr"],
                )
            if resumed_discriminator and override_resume_disc_lr:
                logger.info(
                    "Discriminator LR overridden from the YAML: %.6e",
                    discriminator_optimizer.param_groups[0]["lr"],
                )

    generator_last_layer = get_generator_last_layer(model)

    # ------------------------------------------------------------------
    # DDP
    # ------------------------------------------------------------------
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader, train_sampler = build_dataloader(
        config,
        rank,
        world_size,
        training=True,
    )

    valid_loader = None
    valid_sampler = None

    # 新版Flickr8k划分不再依赖data.valid_path。
    # 可在yaml中通过enable_validation关闭验证。
    enable_validation = bool(
        train_cfg.get("enable_validation", True)
    )

    if enable_validation:
        valid_loader, valid_sampler = build_dataloader(
            config,
            rank,
            world_size,
            training=False,
        )

    if rank == 0:
        total_params = sum(
            parameter.numel()
            for parameter in unwrap_model(model).parameters()
        )

        trainable_params = sum(
            parameter.numel()
            for parameter in main_parameters
        )

        logger.info(
            "Total parameters: %.2f M",
            total_params / 1e6,
        )

        logger.info(
            "Main trainable parameters: %.2f M",
            trainable_params / 1e6,
        )
        logger.info(
            "Discriminator parameters: %.2f M",
            sum(
                parameter.numel()
                for parameter in criterion.discriminator.parameters()
            ) / 1e6,
        )
        logger.info(
            "Adversarial training starts at step %d",
            criterion.disc_start,
        )

        logger.info(
            "Training image-caption pairs: %d",
            len(train_loader.dataset),
        )

        if valid_loader is not None:
            logger.info(
                "Validation images: %d",
                len(valid_loader.dataset),
            )

    # ------------------------------------------------------------------
    # Training settings
    # ------------------------------------------------------------------
    max_steps = int(train_cfg["max_steps"])

    if criterion.disc_start >= max_steps:
        if rank == 0:
            logger.warning(
                "loss.disc_start=%d is not below train.max_steps=%d; "
                "GAN training will never be enabled.",
                criterion.disc_start,
                max_steps,
            )
    elif criterion.disc_start < train_steps:
        if rank == 0:
            logger.warning(
                "loss.disc_start=%d is below resumed step=%d; "
                "GAN training will be active immediately.",
                criterion.disc_start,
                train_steps,
            )
    elif rank == 0:
        stage2_length = max_steps - train_steps
        pre_gan_steps = criterion.disc_start - train_steps
        pre_gan_ratio = (
            pre_gan_steps / stage2_length
            if stage2_length > 0
            else 0.0
        )
        logger.info(
            "Stage-2 schedule: resume_step=%d, max_steps=%d, "
            "GAN starts at step=%d after %d non-GAN stage-2 steps "
            "(%.2f%% of stage 2).",
            train_steps,
            max_steps,
            criterion.disc_start,
            pre_gan_steps,
            100.0 * pre_gan_ratio,
        )

    log_every = int(
        train_cfg.get("log_every", 100)
    )
    ckpt_every = int(
        train_cfg.get("ckpt_every", 5000)
    )
    valid_every = int(
        train_cfg.get("valid_every", log_every)
    )
    ema_decay = float(
        train_cfg.get("ema_decay", 0.999)
    )
    max_grad_norm = optim_cfg.get(
        "max_grad_norm"
    )

    running: Dict[str, float] = {}
    running_aux = 0.0
    running_grad_norm = 0.0
    running_discriminator_grad_norm = 0.0
    running_discriminator: Dict[str, float] = {}
    running_count = 0

    start_time = time()

    model.train()

    try:
        # --------------------------------------------------------------
        # Main training loop
        # --------------------------------------------------------------
        while train_steps < max_steps:
            epoch = train_steps // max(
                len(train_loader),
                1,
            )

            train_sampler.set_epoch(epoch)

            for raw_batch in train_loader:
                if train_steps >= max_steps:
                    break

                batch = parse_batch(
                    raw_batch,
                    device,
                )

                batch = attach_text_embeddings(
                    batch,
                    text_encoder,
                    device,
                )

                (
                    reconstruction,
                    clip_align_loss,
                    distill_loss,
                    y_likelihoods,
                    z_likelihoods,
                ) = model(
                    batch.image,
                    batch.text_emb,
                    batch.img_emb,
                )

                # ------------------------------------------------------
                # Generator / codec update (official optimizer_idx == 0)
                # ------------------------------------------------------
                set_requires_grad(
                    criterion.discriminator,
                    False,
                )

                total_loss, logs = criterion.generator_loss(
                    target=batch.image,
                    reconstruction=reconstruction,
                    clip_align_loss=clip_align_loss,
                    distill_loss=distill_loss,
                    y_likelihoods=y_likelihoods,
                    z_likelihoods=z_likelihoods,
                    global_step=train_steps,
                    last_layer=generator_last_layer,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                total_loss.backward()

                grad_norm_value = 0.0

                if max_grad_norm is not None:
                    grad_norm = (
                        torch.nn.utils.clip_grad_norm_(
                            main_parameters,
                            float(max_grad_norm),
                        )
                    )

                    grad_norm_value = float(
                        grad_norm.detach().item()
                    )

                optimizer.step()
                scheduler.step()

                # ------------------------------------------------------
                # Entropy-bottleneck auxiliary update
                # ------------------------------------------------------
                aux_loss = entropy_aux_loss(
                    unwrap_model(model)
                )

                aux_optimizer.zero_grad(
                    set_to_none=True
                )

                aux_loss.backward()
                aux_optimizer.step()

                # ------------------------------------------------------
                # EMA
                # ------------------------------------------------------
                if rank == 0:
                    update_ema(
                        ema_state,
                        unwrap_model(model),
                        ema_decay,
                    )

                # ------------------------------------------------------
                # Discriminator update (official optimizer_idx == 1)
                # ------------------------------------------------------
                set_requires_grad(
                    criterion.discriminator,
                    True,
                )
                discriminator_optimizer.zero_grad(
                    set_to_none=True
                )

                discriminator_loss, discriminator_logs = (
                    criterion.discriminator_step_loss(
                        target=batch.image,
                        reconstruction=reconstruction,
                        global_step=train_steps,
                    )
                )

                discriminator_grad_norm_value = 0.0
                if criterion.adversarial_enabled(train_steps):
                    discriminator_loss.backward()

                    if max_grad_norm is not None:
                        discriminator_grad_norm = (
                            torch.nn.utils.clip_grad_norm_(
                                criterion.discriminator.parameters(),
                                float(max_grad_norm),
                            )
                        )
                        discriminator_grad_norm_value = float(
                            discriminator_grad_norm.detach().item()
                        )

                    discriminator_optimizer.step()

                # The official loop advances the discriminator scheduler
                # every training step, including the pre-disc_start period.
                discriminator_scheduler.step()

                # ------------------------------------------------------
                # Accumulate logs
                # ------------------------------------------------------
                for name, value in logs.items():
                    running[name] = (
                        running.get(name, 0.0)
                        + value.item()
                    )

                for name, value in discriminator_logs.items():
                    running_discriminator[name] = (
                        running_discriminator.get(name, 0.0)
                        + value.item()
                    )

                running_aux += aux_loss.item()
                running_grad_norm += grad_norm_value
                running_discriminator_grad_norm += (
                    discriminator_grad_norm_value
                )
                running_count += 1
                train_steps += 1

                # ------------------------------------------------------
                # Training logging and TensorBoard
                # ------------------------------------------------------
                if train_steps % log_every == 0:
                    elapsed = max(
                        time() - start_time,
                        1e-6,
                    )

                    local_values = {
                        name: torch.tensor(
                            value / running_count,
                            device=device,
                        )
                        for name, value
                        in running.items()
                    }

                    local_values["aux"] = torch.tensor(
                        running_aux / running_count,
                        device=device,
                    )

                    for name, value in running_discriminator.items():
                        local_values[name] = torch.tensor(
                            value / running_count,
                            device=device,
                        )

                    local_values["grad_norm"] = (
                        torch.tensor(
                            running_grad_norm
                            / running_count,
                            device=device,
                        )
                    )
                    local_values["discriminator_grad_norm"] = (
                        torch.tensor(
                            running_discriminator_grad_norm
                            / running_count,
                            device=device,
                        )
                    )

                    averages = {
                        name: reduce_mean(value).item()
                        for name, value
                        in local_values.items()
                    }

                    current_lr = (
                        optimizer.param_groups[0]["lr"]
                    )

                    steps_per_second = (
                        running_count / elapsed
                    )

                    if rank == 0:
                        logger.info(
                            "step=%07d "
                            "lr=%.6e "
                            "steps/s=%.2f | %s",
                            train_steps,
                            current_lr,
                            steps_per_second,
                            " | ".join(
                                f"{name}={value:.6f}"
                                for name, value
                                in averages.items()
                            ),
                        )

                    if (
                        rank == 0
                        and writer is not None
                    ):
                        for name, value in (
                            averages.items()
                        ):
                            writer.add_scalar(
                                f"train/{name}",
                                value,
                                train_steps,
                            )

                        writer.add_scalar(
                            "train/lr",
                            current_lr,
                            train_steps,
                        )
                        writer.add_scalar(
                            "train/discriminator_lr",
                            discriminator_optimizer.param_groups[0]["lr"],
                            train_steps,
                        )

                        writer.add_scalar(
                            "train/steps_per_second",
                            steps_per_second,
                            train_steps,
                        )

                        writer.add_scalar(
                            "system/gpu_memory_allocated_gb",
                            torch.cuda.memory_allocated(
                                device
                            )
                            / 1024**3,
                            train_steps,
                        )

                        writer.add_scalar(
                            "system/gpu_memory_reserved_gb",
                            torch.cuda.memory_reserved(
                                device
                            )
                            / 1024**3,
                            train_steps,
                        )

                        writer.add_scalar(
                            "system/gpu_memory_peak_gb",
                            torch.cuda.max_memory_allocated(
                                device
                            )
                            / 1024**3,
                            train_steps,
                        )

                    running.clear()
                    running_aux = 0.0
                    running_grad_norm = 0.0
                    running_discriminator_grad_norm = 0.0
                    running_discriminator.clear()
                    running_count = 0
                    start_time = time()

                # ------------------------------------------------------
                # Validation
                # ------------------------------------------------------
                if (
                    valid_loader is not None
                    and train_steps % valid_every == 0
                ):
                    if valid_sampler is not None:
                        valid_sampler.set_epoch(
                            train_steps
                        )

                    metrics = evaluate(
                        model,
                        valid_loader,
                        criterion,
                        device,
                        text_encoder,
                    )

                    if rank == 0:
                        logger.info(
                            "Validation step=%07d | %s",
                            train_steps,
                            " | ".join(
                                f"{name}={value:.6f}"
                                for name, value
                                in metrics.items()
                            ),
                        )

                    if (
                        rank == 0
                        and writer is not None
                    ):
                        for name, value in (
                            metrics.items()
                        ):
                            writer.add_scalar(
                                f"valid/{name}",
                                value,
                                train_steps,
                            )

                        # 验证频率低，验证后立即写入磁盘
                        writer.flush()

                    dist.barrier()

                # ------------------------------------------------------
                # Periodic checkpoint
                # ------------------------------------------------------
                if train_steps % ckpt_every == 0:
                    if rank == 0:
                        checkpoint_path = (
                            checkpoint_dir
                            / f"{train_steps:07d}.pt"
                        )

                        save_checkpoint(
                            checkpoint_path,
                            unwrap_model(model),
                            criterion,
                            ema_state,
                            optimizer,
                            discriminator_optimizer,
                            aux_optimizer,
                            scheduler,
                            discriminator_scheduler,
                            train_steps,
                            config,
                        )

                        logger.info(
                            "Saved checkpoint: %s",
                            checkpoint_path,
                        )

                    dist.barrier()

        # --------------------------------------------------------------
        # Final checkpoint
        # --------------------------------------------------------------
        if rank == 0:
            final_path = (
                checkpoint_dir / "final.pt"
            )

            save_checkpoint(
                final_path,
                unwrap_model(model),
                criterion,
                ema_state,
                optimizer,
                discriminator_optimizer,
                aux_optimizer,
                scheduler,
                discriminator_scheduler,
                train_steps,
                config,
            )

            logger.info(
                "Training completed. "
                "Final checkpoint: %s",
                final_path,
            )

    finally:
        # 即使训练过程中出现异常，也尽量把已有TensorBoard数据刷盘
        if rank == 0 and writer is not None:
            writer.flush()
            writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    rank, world_size, local_rank = setup_ddp()

    try:
        set_seed(int(config["train"].get("global_seed", 903)), rank)
        train(rank, world_size, local_rank, config)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
