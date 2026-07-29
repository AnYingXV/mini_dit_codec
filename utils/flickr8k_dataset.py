from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class Flickr8kSingleCaption(Dataset):
    def __init__(self, image_root, annotation_file, transform=None, caption_index=0):
        self.image_root = Path(image_root)
        self.transform = transform
        self.caption_index = caption_index
        self.samples = []

        captions = {}

        with open(annotation_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()

                if not line or "\t" not in line:
                    continue

                image_id, caption = line.split("\t", 1)

                if "#" not in image_id:
                    continue

                image_name, caption_id = image_id.rsplit("#", 1)

                try:
                    caption_id = int(caption_id)
                except ValueError:
                    continue

                captions.setdefault(image_name, {})[caption_id] = caption

        for image_name, image_captions in captions.items():
            image_path = self.image_root / image_name

            if image_path.exists() and self.caption_index in image_captions:
                self.samples.append((image_path, image_captions[self.caption_index]))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid Flickr8k samples found: "
                f"image_root={image_root}, annotation_file={annotation_file}"
            )

        print(f"Loaded Flickr8k samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, caption = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "caption": caption,
        }
