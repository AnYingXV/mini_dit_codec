import os
import math
import glob
import torch
import torch.nn.functional as F

from PIL import Image
from DiT_IC import DiT_IC
from torchvision import transforms
from utils_modules.compress_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 单张图片加载和预处理
def load_img(img_path, transform):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img)
    return img_tensor


# 单图片编码√
def compress_one_img(net, img, bin_path, img_name, ori_h, ori_w):
    img_bin = os.path.join(bin_path, img_name + ".bin")
    with torch.no_grad(): 
        compress_dict = net.compress(img)

    with open(img_bin, "wb") as f:
        write_body(f, 
                    compress_dict["z_shape"], 
                    compress_dict["strings"]
                    ) 
    size = os.path.getsize(img_bin)
    bpp = float(size)*8 / float(ori_h*ori_w)

    return bpp, img_bin
    
# 单图片解码
def decompress_one_img(net, ori_h, ori_w, img_bin):
    with open(img_bin, "rb") as f:
        strings, z_shape = read_body(f)
    with torch.no_grad():
        out_img = net.decompress(strings, z_shape)
    out_img = out_img[:, :, 0 : ori_h, 0 : ori_w]
    out_img = (out_img * 0.5 + 0.5).float().clamp(0.0, 1.0)
    return out_img


def main(dataset_path, bin_path, out_path):
    imgs_path = glob.glob(dataset_path + "/*.png")
    print(f"Find total: {len(imgs_path)} images.")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    net = DiT_IC()
    net = net.to(device).eval()
    net.latent_codec.update(force=True)

    # 逐张进行
    for img_path in imgs_path:
        _, img_name_ex = os.path.split(img_path) # 返回 目录、含后缀文件名
        name, _ = os.path.splitext(img_name_ex)
        img = load_img(img_path, transform).unsqueeze(0).to(device)

        # pad
        _, _, ori_h, ori_w = img.shape
        h_pad = math.ceil(ori_h/256)*256 - ori_h # ceil向上取整
        w_pad = math.ceil(ori_w/256)*256 - ori_w
        img_pad = F.pad(img,(0, w_pad, 0, h_pad), mode='reflect')

        
        # compress
        bpp, img_bin = compress_one_img(net, img_pad, bin_path, name, ori_h, ori_w)
        print(f"{name}: {bpp:.4f} bpp")

        # decompress
        out_img = decompress_one_img(net, ori_h, ori_w, img_bin)
    
        save_path = os.path.join(out_path, name+'.png')
        output_img = transforms.ToPILImage()(out_img[0].cpu()) 
        output_img.save(save_path)




if __name__ == '__main__':
    dataset_path =  "/datasets/Kodak/"
    bin_path = "/img_research/mini_dit_codec/output/bin/"


