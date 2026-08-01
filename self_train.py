import os
import math
import torch
import pyiqa
import argparse
import diffusers
import numpy as np
import transformers

from torch.utils.data import DataLoader
from accelerate.utils import set_seed
from diffusers.utils.import_utils import is_xformers_available
from model.DiT_IC import DiT_IC
from tqdm.auto import tqdm
from model.losses import DiTICLosses
from accelerate import Accelerator
from torchvision import transforms

from transformers import get_scheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

from utils.flickr8k_dataset import Flickr8kSingleCaption
from compressai.datasets import ImageFolder
from transformers import CLIPModel, CLIPProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Train DiT-IC image compression model.")

    # ------------------------- 预训练模型 -------------------------
    parser.add_argument("--clip_path", type=str, required=True, help="CLIP预训练模型路径或HF模型名。")
    parser.add_argument("--dit_path", type=str, required=True, help="DiT预训练模型路径。")
    parser.add_argument("--elic_path", type=str, required=True, help="ELIC预训练模型路径。")

    # ------------------------- Accelerator -------------------------
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to", type=str, default="tensorboard", choices=["tensorboard", "wandb", "comet_ml", "all", "none"])

    # ------------------------- 数据集 -------------------------
    parser.add_argument("--train_image_root", type=str, required=True)
    parser.add_argument("--train_caption_file", type=str, required=True)
    parser.add_argument("--test_dataset", type=str, required=True)
    parser.add_argument("--train_patch_size", type=int, default=256)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)

    # ------------------------- 显存优化 -------------------------
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    # ------------------------- 训练 -------------------------
    parser.add_argument("--max_train_steps", type=int, default=100000)
    parser.add_argument("--train_stage", type=int, default=1, choices=[1, 2])
    parser.add_argument("--checkpointing_steps", type=int, default=5000)
    parser.add_argument("--eval_freq", type=int, default=1000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # ------------------------- Loss -------------------------
    parser.add_argument("--lambda_rate", type=float, default=1.0)
    parser.add_argument("--lambda_mse", type=float, default=1.0)
    parser.add_argument("--lambda_lpips", type=float, default=1.0)
    parser.add_argument("--lambda_dists", type=float, default=1.0)
    parser.add_argument("--lambda_distill", type=float, default=1.0)
    parser.add_argument("--lambda_cond", type=float, default=1.0)
    parser.add_argument("--lambda_adv", type=float, default=0.0)

    # ------------------------- 输出 -------------------------
    parser.add_argument("--output_path", type=str, required=True)

    args = parser.parse_args()

    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient_accumulation_steps must be >= 1.")
    if args.train_batch_size < 1:
        parser.error("--train_batch_size must be >= 1.")
    if args.train_patch_size < 1:
        parser.error("--train_patch_size must be >= 1.")
    if args.max_train_steps < 1:
        parser.error("--max_train_steps must be >= 1.")
    if args.checkpointing_steps < 1:
        parser.error("--checkpointing_steps must be >= 1.")
    if args.eval_freq < 1:
        parser.error("--eval_freq must be >= 1.")

    return args


def main(args):

    #------------------------------------ 先创建好Accelerator，然后设置一些最基本的参数 ---------------------------------------------

    # 加载CLIP
    clip_model = CLIPModel.from_pretrained(args.clip_path)
    clip_processor = CLIPProcessor.from_pretrained(args.clip_path)
    clip_model.requires_grad_(False)
    clip_model.eval()


    # 训练包装器
    accelerator = Accelerator(
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        mixed_precision = args.mixed_precision, # 设置整个训练过程的计算精度
        log_with=args.report_to,
        project_dir=args.output_path,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers("DiT-IC")


    # 设置日志输出
    if accelerator.is_local_main_process: # 判断是否是本机主进程 rank0一般是主进程
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # 设置随机种子
    if args.seed is not None:
        set_seed(args.seed)

    #----------------------------------------------------- 处理数据 -------------------------------------------------------------------------

    # 数据增强，使用Stablecodec的增强方法
    train_transform = transforms.Compose([
        transforms.Resize(args.train_patch_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop((args.train_patch_size, args.train_patch_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    train_dataset = Flickr8kSingleCaption(
        image_root=args.train_image_root,
        annotation_file=args.train_caption_file,
        transform=train_transform,
        caption_index=0,
    )
    print("train_dataset =", len(train_dataset))

    test_dataset = ImageFolder(
        args.test_dataset,
        split="Kodak",
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    )
    print("test_dataset =", len(test_dataset))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
        shuffle=False,
        pin_memory=True,
    )

    #-------------------------------------------------- 提高硬件计算效率的设置：省显存、提速 -------------------------------------------------

    net = DiT_IC(args.dit_path, args.elic_path)
    net.train()

    # 创建训练输出文件夹
    if accelerator.is_main_process: # 主进程创建即可
        os.makedirs(os.path.join(args.output_path, "checkpoints"), exist_ok = True)
        os.makedirs(os.path.join(args.output_path, "eval"), exist_ok=True)

    # 显存优化方法1：针对注意力机制
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net.DiT.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    # 显存优化方法2：针对中间激活层的输出
    if args.gradient_checkpointing:
        net.DiT.enable_gradient_checkpointing()
    '''
    为了能够反向传播，普通训练的时候，前向传播时每层的输出都会保存，这部分很消耗显存。
    gradient_checkpointing的解决思路是前向传播的时候不保存各层输出, 反向传播的时候, 传到哪层, 再前向计算到那层一遍。
    缺点：耗时变慢
    '''

    # 加速方法：tf32精度
    torch.backends.cuda.matmul.allow_tf32 = True


    #---------------------------------------------------------- 将所有要训练的网络传给optimizer，设置device、精度、优化器，创建loss失真项所需的MSE/DISTS/LPIPS -----------------------------------------------------------

    MSE_loss = torch.nn.MSELoss()

    LPIPS_loss = pyiqa.create_metric("lpips", as_loss=True)
    LPIPS_loss.requires_grad_(False)
    LPIPS_loss.eval()

    DISTS_loss = pyiqa.create_metric("dists", as_loss=True)
    DISTS_loss.requires_grad_(False)
    DISTS_loss.eval()

    trainable_params = [param for param in net.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr = 1e-4, betas = (0.9,0.999), weight_decay = 0.0, eps = 1e-8)
    lr_scheduler = get_scheduler(
        name="constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=700,
        num_training_steps=args.max_train_steps,
    )

    net, clip_model, optimizer, train_dataloader, LPIPS_loss, DISTS_loss, lr_scheduler = accelerator.prepare(net, clip_model, optimizer, train_dataloader, LPIPS_loss, DISTS_loss, lr_scheduler) # 把上面设置的所有训练用到的，全都丢给accelerator.prepare，他会自动设置/协调好和硬件的协作


    #------------------------------------------------------------------ 训练loop ---------------------------------------------------------------

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps", disable=not accelerator.is_local_main_process,)
    Loss = DiTICLosses(args.lambda_mse, args.lambda_lpips, args.lambda_dists, args.lambda_distill, args.lambda_cond, args.lambda_adv)
    train_steps = 0
    while train_steps < args.max_train_steps:
        for batch in train_dataloader:

            # 获取CLIP对齐文本
            images = batch["image"]
            captions = batch["caption"]
            with torch.no_grad():
                clip_inputs = clip_processor(
                    text=list(captions),
                    # images=(images * 0.5 + 0.5).clamp(0, 1),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                clip_inputs = {
                    k: v.to(accelerator.device)
                    for k, v in clip_inputs.items()
                }
                text_emb = clip_model.get_text_features(
                    input_ids=clip_inputs["input_ids"],
                    attention_mask=clip_inputs["attention_mask"],
                )

                # img_emb = clip_model.get_image_features(
                #     pixel_values=clip_inputs["pixel_values"],
                # )


            with accelerator.accumulate(net):
                
                # 获取模型输出
                b, c, h, w = images.shape
                x_hat, clip_align_loss, distill_loss, y_likelihoods, z_likelihoods = net(images, text_emb)
                x_hat = x_hat.float()
                x = images.float()

                # 计算损失函数
                num_pixels = b*h*w
                mse_loss = MSE_loss(x_hat, x)
                x_01 = (x * 0.5 + 0.5).clamp(0, 1)
                x_hat_01 = (x_hat * 0.5 + 0.5).clamp(0, 1)
                lpips_loss = LPIPS_loss(x_hat_01.contiguous(), x_01.contiguous()).mean()
                dists_loss = DISTS_loss(x_hat_01.contiguous(), x_01.contiguous()).mean()
                total_loss = Loss.stage1(y_likelihoods, z_likelihoods, num_pixels, mse_loss, lpips_loss, dists_loss, distill_loss, clip_align_loss, args.lambda_rate)
                
                # 参数更新+梯度剪裁
                accelerator.backward(total_loss) # 反向算梯度
                if accelerator.sync_gradients: # 判断当前步是否是参数更新步
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm) # 在参数更新步执行梯度剪裁
                optimizer.step() # 更新参数
                lr_scheduler.step()
                with torch.no_grad():
                    raw_net = accelerator.unwrap_model(net)
                    raw_net.prompter.logit_scale.clamp_(
                        max=math.log(100.0)
                    )
                optimizer.zero_grad(set_to_none=True) # 梯度清零


            if accelerator.sync_gradients:
                progress_bar.update(1)
                train_steps += 1

                train_logs = {
                    "train/loss": total_loss.detach().item(),
                    "train/mse": mse_loss.detach().item(),
                    "train/lpips": lpips_loss.detach().item(),
                    "train/dists": dists_loss.detach().item(),
                }
                accelerator.log(train_logs, step=train_steps)
                accelerator.log({
                    "train/logit_scale":
                        accelerator.unwrap_model(net).prompter.logit_scale.item(),

                    "train/logit_scale_exp":
                        accelerator.unwrap_model(net).prompter.logit_scale.exp().item(),
                }, step=train_steps)


                if accelerator.is_main_process:

                    # 保存 checkpoint
                    if train_steps % args.checkpointing_steps == 0:
                        save_path = os.path.join(
                            args.output_path,
                            "checkpoints",
                            f"ditic_stage1_{train_steps}.pth",
                        )

                        torch.save(
                            {
                                "step": train_steps,
                                "model": accelerator.unwrap_model(net).state_dict(),
                                "optimizer": optimizer.state_dict(),
                            },
                            save_path,
                        )

                    # 验证
                    if train_steps % args.eval_freq == 0:
                        net.eval()

                        val_rate = []
                        val_psnr = []
                        val_lpips = []
                        save_count = 0

                        for idx, batch_val in enumerate(test_dataloader):
                            batch_val = batch_val.to(accelerator.device, non_blocking=True)

                            with torch.no_grad():
                                x_hat_val, _, _, y_likelihoods, z_likelihoods = accelerator.unwrap_model(net)(batch_val, text_emb=None)

                                B, _, H, W = batch_val.shape
                                num_pixels = B * H * W

                                y_bpp = -torch.log2(y_likelihoods.float().clamp_min(1e-9)).sum() / num_pixels
                                z_bpp = -torch.log2(z_likelihoods.float().clamp_min(1e-9)).sum() / num_pixels
                                total_bpp = y_bpp + z_bpp

                                # LPIPS
                                x_val = batch_val.float()
                                x_hat_val = x_hat_val.float()

                                x_01 = (x_val * 0.5 + 0.5).clamp(0, 1)
                                x_hat_01 = (x_hat_val * 0.5 + 0.5).clamp(0, 1)

                                lpips_score = LPIPS_loss(x_hat_01.contiguous(), x_01.contiguous()).mean()

                                # PSNR
                                mse_val = torch.nn.functional.mse_loss(x_hat_01, x_01)
                                psnr = -10.0 * torch.log10(mse_val.clamp_min(1e-10))

                            val_rate.append(total_bpp.item())
                            val_psnr.append(psnr.item())
                            val_lpips.append(lpips_score.item())

                            if save_count < 10:
                                comparison = torch.cat([x_01.cpu(), x_hat_01.cpu()], dim=3)
                                comparison_image = transforms.ToPILImage()(comparison[0])
                                comparison_image.save(os.path.join(args.output_path, "eval", f"step_{train_steps}_{idx}.png"))
                                save_count += 1

                        val_logs = {
                            "val/bpp": float(np.mean(val_rate)),
                            "val/psnr": float(np.mean(val_psnr)),
                            "val/lpips": float(np.mean(val_lpips)),
                        }

                        progress_bar.set_postfix(**val_logs)
                        accelerator.log(val_logs, step=train_steps)

                        net.train()

                if train_steps >= args.max_train_steps:
                    accelerator.wait_for_everyone()
                    break


if __name__ == "__main__":
    args = parse_args()
    main(args)
