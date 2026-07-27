import torch
import diffusers
import transformers

from accelerate import Accelerator
from torchvision import transforms
from utils.training_utils import H5Dataset
from compressai.datasets import ImageFolder


def main(args):

    #------------------------------------ 先创建好Accelerator，然后设置一些最基本的参数 ---------------------------------------------

    # 加载基模
    dit_path = args.dit_path

    # 训练包装器
    accelerator = Accelerator(
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        mixed_precision = args.mixed_precision, # 设置整个训练过程的计算精度
        log_with=args.report_to,
    )

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

    # 使用HDF5文件的方法组织数据集
    # 数据增强，使用Stablecodec的增强方法
    train_dataset = H5Dataset(
        args.train_dataset,
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop((args.train_patch_size, args.train_patch_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    )

    test_dataset = ImageFolder(
        args.test_dataset,
        split="Kodak",
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    )

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
    net.set_train()

    # 创建训练输出文件夹
    if accelerator.is_main_process: # 主进程创建即可
        os.makedirs(os.path.join(args.output_path, "checkpoints"), exist_ok = True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    # 显存优化方法1：针对注意力机制
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net.dit.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    # 显存优化方法2：针对中间激活层的输出
    if args.gradient_checkpointing:
        net.dit.enable_gradient_checkpointing()
    '''
    为了能够反向传播，普通训练的时候，前向传播时每层的输出都会保存，这部分很消耗显存。
    gradient_checkpointing的解决思路是前向传播的时候不保存各层输出, 反向传播的时候, 传到哪层, 再前向计算到那层一遍。
    缺点：耗时变慢
    '''

    # 加速方法：tf32精度
    torch.backends.cuda.matmul.allow_tf32 = True


    #---------------------------------------------------------- 将所有要训练的网络传给optimizer，设置device、精度、优化器，创建loss失真项所需的MSE/DISTS/LPIPS -----------------------------------------------------------

    trainable_params = [param for param in net.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr = 1e-4, betas = (0.9,0.999), weight_decay = 0.01, eps = 1e-8)
    net, optimizer, train_dataloader = accelerator .prepare(net, optimizer, train_dataloader) # 把上面设置的所有训练用到的，全都丢给accelerator.prepare，他会自动设置/协调好和硬件的协作


    mse_loss = torch.nn.MSELoss()

    lpips_loss = lpips.LPIPS(net = "vgg")
    lpips_loss.requires_grad_(False)
    lpips_loss.eval()

    dists_loss = DISTS()
    dists_loss.requires_grad_(False)
    dists_loss.eval()


    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    lpips_loss.to(accelerator.device, dtype=weight_dtype)
    dists_loss.to(accelerator.device, dtype=weight_dtype)
    net.to(accelerator.device)


    #------------------------------------------------------------------ 训练loop ---------------------------------------------------------------

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps", disable=not accelerator.is_local_main_process,)
    train_steps = 0

    
