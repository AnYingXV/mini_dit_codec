"""
DiT-IC inference/evaluation script.

Metrics are intentionally matched to train_ditic_ddp_flickr8k_split.py:
  - PSNR: -10 * log10(MSE), evaluated on tensors in [-1, 1]
  - LPIPS: lpips.LPIPS, evaluated on tensors in [-1, 1]
  - DISTS: DISTS_pytorch.DISTS, evaluated on tensors converted to [0, 1]
  - BPP: either real bitstream size or entropy-likelihood estimation
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from DISTS_pytorch import DISTS
from PIL import Image
from torch import Tensor
from torchvision import transforms

from eval.compress_utils import filesize, read_body, write_body
from model.DiT_IC import DiT_IC


IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")


class LPIPSLoss(nn.Module):
    """Same LPIPS wrapper used by the training script."""

    def __init__(self, net: str = "alex") -> None:
        super().__init__()
        self.metric = lpips.LPIPS(net=net)
        self.metric.eval()
        self.metric.requires_grad_(False)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        # lpips expects images in [-1, 1].
        return self.metric(prediction, target).mean()


class DISTSLoss(nn.Module):
    """Same DISTS wrapper and input conversion used by the training script."""

    def __init__(self) -> None:
        super().__init__()
        self.metric = DISTS()
        self.metric.eval()
        self.metric.requires_grad_(False)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        # The training code converts [-1, 1] to [0, 1] for DISTS.
        prediction_01 = (prediction + 1.0) * 0.5
        target_01 = (target + 1.0) * 0.5
        return self.metric(prediction_01, target_01).mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compress, reconstruct, and evaluate images with DiT-IC."
    )
    parser.add_argument(
        "--config",
        "--config_path",
        dest="config_path",
        type=str,
        required=True,
        help="Training YAML containing model.dit_path and model.elic_path.",
    )
    parser.add_argument(
        "--checkpoint",
        "--codec_path",
        dest="checkpoint_path",
        type=str,
        required=True,
        help="Checkpoint produced by train_ditic_ddp_flickr8k_split.py.",
    )
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--rec_path", type=str, default="./reconstructions")
    parser.add_argument("--bin_path", type=str, default="./bitstreams")
    parser.add_argument("--seed", type=int, default=903)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='For example "cuda", "cuda:0", or "cpu".',
    )
    parser.add_argument(
        "--use_ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load checkpoint['ema'] instead of checkpoint['model'] when available.",
    )
    parser.add_argument(
        "--entropy_estimation",
        action="store_true",
        help="Estimate BPP from likelihoods instead of writing a real bitstream.",
    )
    parser.add_argument(
        "--save_img",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--padding_factor",
        type=int,
        default=256,
        help="Pad height and width to a multiple of this value.",
    )
    parser.add_argument(
        "--lpips_net",
        type=str,
        default="alex",
        choices=("alex", "vgg", "squeeze"),
        help="LPIPS backbone. Use the same value as loss.lpips_net in training.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("The YAML root must be a mapping.")
    return config


def normalize_state_dict_keys(
    state_dict: Mapping[str, Tensor],
) -> Dict[str, Tensor]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return dict(state_dict)


def load_model(
    config: Mapping[str, Any],
    checkpoint_path: str,
    device: torch.device,
    use_ema: bool,
) -> DiT_IC:
    model_cfg = config.get("model")
    if not isinstance(model_cfg, Mapping):
        raise KeyError("Config must contain a 'model' mapping.")

    try:
        dit_path = model_cfg["dit_path"]
        elic_path = model_cfg["elic_path"]
    except KeyError as exc:
        raise KeyError(
            "Config model section must contain 'dit_path' and 'elic_path'."
        ) from exc

    net = DiT_IC(dit_path=dit_path, elic_path=elic_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a mapping.")

    if use_ema and "ema" in checkpoint:
        state_dict = checkpoint["ema"]
        state_name = "EMA"
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
        state_name = "model"
    else:
        state_dict = checkpoint
        state_name = "plain state_dict"

    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Selected {state_name} weights are not a state_dict mapping.")

    state_dict = normalize_state_dict_keys(state_dict)
    incompatible = net.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint does not exactly match DiT_IC.\n"
            f"Missing keys ({len(incompatible.missing_keys)}): "
            f"{incompatible.missing_keys[:20]}\n"
            f"Unexpected keys ({len(incompatible.unexpected_keys)}): "
            f"{incompatible.unexpected_keys[:20]}"
        )

    net.to(device).eval()

    # Required by CompressAI before real entropy coding.
    net.latent_codec.update(force=True)

    print(f"Loaded {state_name} weights from: {checkpoint_path}")
    return net


def preprocess_image(image_path: str, transform: transforms.Compose) -> Tensor:
    with Image.open(image_path) as image:
        return transform(image.convert("RGB"))


def safe_pad_to_factor(x: Tensor, factor: int) -> Tuple[Tensor, int, int]:
    if factor <= 0:
        raise ValueError("padding_factor must be positive.")

    height, width = x.shape[-2:]
    pad_h = math.ceil(height / factor) * factor - height
    pad_w = math.ceil(width / factor) * factor - width
    if pad_h == 0 and pad_w == 0:
        return x, 0, 0

    mode = "reflect" if pad_h < height and pad_w < width else "replicate"
    return F.pad(x, (0, pad_w, 0, pad_h), mode=mode), pad_h, pad_w


def compress_one_image(
    net: DiT_IC,
    bin_path: str,
    ori_h: int,
    ori_w: int,
    img_name: str,
    x: Tensor,
) -> float:
    with torch.inference_mode():
        output_dict = net.compress(x)

    if "strings" not in output_dict or "z_shape" not in output_dict:
        raise KeyError(
            "DiT_IC.compress() must return keys 'strings' and 'z_shape'."
        )

    strings = output_dict["strings"]
    z_shape = output_dict["z_shape"]

    Path(bin_path).mkdir(parents=True, exist_ok=True)
    output_path = Path(bin_path) / f"{img_name}.bin"
    with output_path.open("wb") as file:
        write_body(file, z_shape, strings)

    return float(filesize(output_path)) * 8.0 / float(ori_h * ori_w)


def decompress_one_image(
    net: DiT_IC,
    bin_path: str,
    ori_h: int,
    ori_w: int,
    img_name: str,
) -> Tensor:
    input_path = Path(bin_path) / f"{img_name}.bin"
    with input_path.open("rb") as file:
        strings, z_shape = read_body(file)

    with torch.inference_mode():
        reconstruction = net.decompress(strings, z_shape)

    # Keep metric tensors in [-1, 1], exactly as in training.
    return reconstruction[..., :ori_h, :ori_w].float().clamp(-1.0, 1.0)


def estimate_one_image(
    net: DiT_IC,
    x_padded: Tensor,
    ori_h: int,
    ori_w: int,
) -> Tuple[Tensor, Tensor]:
    # Match training: number of pixels excludes channels and uses original image size.
    num_pixels = x_padded.new_tensor(float(x_padded.shape[0] * ori_h * ori_w))

    with torch.inference_mode():
        (
            reconstruction,
            _clip_align_loss,
            _distill_loss,
            y_likelihoods,
            z_likelihoods,
        ) = net(x_padded)

    likelihoods: Iterable[Tensor] = (y_likelihoods, z_likelihoods)
    bpp = sum(
        torch.log(likelihood.clamp_min(1e-9)).sum()
        / (-math.log(2.0) * num_pixels)
        for likelihood in likelihoods
    )

    reconstruction = reconstruction[..., :ori_h, :ori_w]
    return reconstruction.float().clamp(-1.0, 1.0), bpp


def compute_metrics(
    reconstruction: Tensor,
    target: Tensor,
    lpips_metric: LPIPSLoss,
    dists_metric: DISTSLoss,
) -> Dict[str, float]:
    # This duplicates the training criterion's PSNR definition.
    mse = F.mse_loss(reconstruction, target)
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))

    with torch.inference_mode():
        lpips_value = lpips_metric(reconstruction, target)
        dists_value = dists_metric(reconstruction, target)

    return {
        "psnr": float(psnr.item()),
        "lpips": float(lpips_value.item()),
        "dists": float(dists_value.item()),
    }


def collect_images(img_path: str) -> list[str]:
    images: list[str] = []
    for extension in IMAGE_EXTENSIONS:
        images.extend(glob.glob(os.path.join(img_path, extension)))
        images.extend(glob.glob(os.path.join(img_path, extension.upper())))
    return sorted(set(images))


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is unavailable.")

    config = load_config(args.config_path)
    net = load_model(
        config=config,
        checkpoint_path=args.checkpoint_path,
        device=device,
        use_ema=args.use_ema,
    )

    lpips_net = args.lpips_net
    loss_cfg = config.get("loss")
    if isinstance(loss_cfg, Mapping):
        lpips_net = str(loss_cfg.get("lpips_net", lpips_net))

    lpips_metric = LPIPSLoss(net=lpips_net).to(device)
    dists_metric = DISTSLoss().to(device)

    if args.save_img:
        Path(args.rec_path).mkdir(parents=True, exist_ok=True)
    if not args.entropy_estimation:
        Path(args.bin_path).mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ]
    )

    images = collect_images(args.img_path)
    if not images:
        raise FileNotFoundError(f"No supported images were found in: {args.img_path}")

    print(f"\nFound {len(images)} images in {args.img_path}\n")

    sums = {"psnr": 0.0, "lpips": 0.0, "dists": 0.0, "bpp": 0.0}

    for img_path in images:
        fname = Path(img_path).stem
        output_path = Path(args.rec_path) / f"{fname}.png"

        target = preprocess_image(img_path, transform).unsqueeze(0).to(device)
        ori_h, ori_w = target.shape[-2:]
        padded, _, _ = safe_pad_to_factor(target, args.padding_factor)

        if args.entropy_estimation:
            reconstruction, bpp_tensor = estimate_one_image(
                net, padded, ori_h, ori_w
            )
            bpp = float(bpp_tensor.item())
        else:
            bpp = compress_one_image(
                net, args.bin_path, ori_h, ori_w, fname, padded
            )
            reconstruction = decompress_one_image(
                net, args.bin_path, ori_h, ori_w, fname
            )

        values = compute_metrics(
            reconstruction=reconstruction,
            target=target,
            lpips_metric=lpips_metric,
            dists_metric=dists_metric,
        )
        values["bpp"] = bpp

        print(f"============== {fname} ==============")
        print(f"PSNR : {values['psnr']:.6f}")
        print(f"LPIPS: {values['lpips']:.6f}")
        print(f"DISTS: {values['dists']:.6f}")
        print(f"BPP  : {values['bpp']:.6f}")

        for key in sums:
            sums[key] += values[key]

        if args.save_img:
            reconstruction_01 = ((reconstruction + 1.0) * 0.5).clamp(0.0, 1.0)
            transforms.ToPILImage()(reconstruction_01[0].cpu()).save(output_path)

    count = len(images)
    print(f"\n============== Average over {count} images ==============")
    print(f"PSNR : {sums['psnr'] / count:.6f}")
    print(f"LPIPS: {sums['lpips'] / count:.6f}")
    print(f"DISTS: {sums['dists'] / count:.6f}")
    print(f"BPP  : {sums['bpp'] / count:.6f}")


if __name__ == "__main__":
    main(parse_args())
