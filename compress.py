import math
import glob
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from DiT_IC import DiT_IC


# 单张图片加载和预处理
def load_img(img_path, transform):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img)
    return img_tensor

def compress_one_img(net, img)

def main(dataset_path):
    imgs_path = glob.glob(dataset_path + "/*.png")
    print(f"Find total: {len(imgs_path)} images.")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    net = DiT_IC()
    net.eval().cuda()

    for img_path in imgs_path:
        img = load_img(img_path, transform).unsqueeze(0)

        # pad
        _, _, h, w = img.shape
        h_pad = math.ceil(h/256)*256 - h # ceil向上取整
        w_pad = math.ceil(w/256)*256 - w
        img = F.pad(img,(0, w_pad, 0, h_pad), mode='reflect')


