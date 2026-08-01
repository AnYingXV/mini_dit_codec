from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset


def _read_flickr8k_annotations(
    annotation_file: str | Path,
) -> Dict[str, Dict[int, str]]:
    """Read Flickr8k token format: image.jpg#0<TAB>caption."""
    captions: Dict[str, Dict[int, str]] = {}

    with open(annotation_file, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            line = line.strip()
            if not line or "\t" not in line:
                continue

            image_id, caption = line.split("\t", 1)
            if "#" not in image_id:
                continue

            image_name, caption_id_text = image_id.rsplit("#", 1)
            try:
                caption_id = int(caption_id_text)
            except ValueError:
                continue

            captions.setdefault(image_name, {})[caption_id] = caption

    if not captions:
        raise RuntimeError(
            f"No valid Flickr8k captions were found in {annotation_file}."
        )
    return captions


def _read_split_file(split_file: str | Path) -> List[str]:
    """Read one image filename per line."""
    with open(split_file, "r", encoding="utf-8", errors="ignore") as file:
        names = [line.strip() for line in file if line.strip()]

    if not names:
        raise RuntimeError(f"Split file is empty: {split_file}")
    return names


def _deterministic_split(
    image_names: Sequence[str],
    split: str,
    validation_ratio: float,
    split_seed: int,
) -> List[str]:
    """
    Deterministically divide image identities into train/validation subsets.

    The split happens before captions are expanded, so all five captions of
    one image stay in the same subset and no image leaks across the split.
    """
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be strictly between 0 and 1.")

    names = sorted(set(image_names))
    if len(names) < 2:
        raise RuntimeError("At least two valid images are required for a split.")

    rng = random.Random(split_seed)
    rng.shuffle(names)

    valid_count = max(1, int(round(len(names) * validation_ratio)))
    valid_count = min(valid_count, len(names) - 1)

    valid_names = names[:valid_count]
    train_names = names[valid_count:]

    if split == "train":
        return train_names
    if split in {"valid", "val", "validation"}:
        return valid_names
    raise ValueError("split must be `train` or `valid`.")


class Flickr8kDataset(Dataset):
    """
    Flickr8k dataset with leakage-free train/validation partitioning.

    Training mode:
      - returns {"image": Tensor, "caption": str}
      - can expand each image into several samples using caption_indices

    Validation mode:
      - returns {"image": Tensor}
      - returns one sample per image and does not expose captions

    Split selection:
      1. If split_file is supplied, filenames are read from that file.
      2. Otherwise a deterministic split is generated with validation_ratio
         and split_seed.
    """

    def __init__(
        self,
        image_root: str | Path,
        annotation_file: str | Path,
        split: str,
        transform=None,
        caption_indices: Sequence[int] = (0, 1, 2, 3, 4),
        return_caption: Optional[bool] = None,
        split_file: Optional[str | Path] = None,
        validation_ratio: float = 0.125,
        split_seed: int = 903,
    ) -> None:
        super().__init__()

        self.image_root = Path(image_root)
        self.annotation_file = Path(annotation_file)
        self.transform = transform
        self.split = "valid" if split in {"valid", "val", "validation"} else split

        if self.split not in {"train", "valid"}:
            raise ValueError("split must be `train` or `valid`.")

        if return_caption is None:
            return_caption = self.split == "train"
        self.return_caption = bool(return_caption)

        captions = _read_flickr8k_annotations(self.annotation_file)

        existing_names = [
            image_name
            for image_name in captions
            if (self.image_root / image_name).is_file()
        ]
        if not existing_names:
            raise RuntimeError(
                "No annotation entries match files under "
                f"image_root={self.image_root}."
            )

        if split_file is not None:
            allowed = set(_read_split_file(split_file))
            selected_names = [
                name for name in sorted(existing_names) if name in allowed
            ]
        else:
            selected_names = _deterministic_split(
                image_names=existing_names,
                split=self.split,
                validation_ratio=validation_ratio,
                split_seed=split_seed,
            )

        if not selected_names:
            raise RuntimeError(
                f"No images remain for split={self.split}. "
                "Check image_root, annotation_file, and split settings."
            )

        self.samples: List[Tuple[Path, Optional[str]]] = []

        if self.return_caption:
            indices = tuple(int(index) for index in caption_indices)
            if not indices:
                raise ValueError(
                    "caption_indices must not be empty when return_caption=True."
                )

            for image_name in selected_names:
                image_captions = captions[image_name]
                image_path = self.image_root / image_name
                for caption_index in indices:
                    caption = image_captions.get(caption_index)
                    if caption is not None:
                        self.samples.append((image_path, caption))
        else:
            # Validation contains each image exactly once and no caption.
            self.samples = [
                (self.image_root / image_name, None)
                for image_name in selected_names
            ]

        if not self.samples:
            raise RuntimeError(
                f"No usable samples found for split={self.split}."
            )

        unique_images = len({path.name for path, _ in self.samples})
        print(
            f"Loaded Flickr8k split={self.split}: "
            f"{unique_images} images, {len(self.samples)} samples, "
            f"return_caption={self.return_caption}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, caption = self.samples[index]

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        sample = {"image": image}
        if self.return_caption:
            if caption is None:
                raise RuntimeError(
                    "A training sample unexpectedly has no caption."
                )
            sample["caption"] = caption
        return sample


class Flickr8kSingleCaption(Flickr8kDataset):
    """
    Backward-compatible wrapper for the original utility.

    It uses the full annotation set unless a split_file or generated split is
    explicitly requested. New code should prefer Flickr8kDataset.
    """

    def __init__(
        self,
        image_root,
        annotation_file,
        transform=None,
        caption_index=0,
        split="train",
        split_file=None,
        validation_ratio=0.125,
        split_seed=903,
        return_caption=True,
    ):
        super().__init__(
            image_root=image_root,
            annotation_file=annotation_file,
            split=split,
            transform=transform,
            caption_indices=(caption_index,),
            return_caption=return_caption,
            split_file=split_file,
            validation_ratio=validation_ratio,
            split_seed=split_seed,
        )
