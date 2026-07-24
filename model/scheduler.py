'''
创新点——自适应步长
'''
import torch
import torch.nn as nn
from diffusers import DPMSolverMultistepScheduler

def make_1step_sched(pretrained_path, device='cuda'):
    noise_scheduler_1step = DPMSolverMultistepScheduler.from_pretrained(pretrained_path, subfolder="scheduler")
    noise_scheduler_1step.set_timesteps(1, device=device)
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.to(device)
    noise_scheduler_1step.betas = noise_scheduler_1step.betas.to(device)

    return noise_scheduler_1step

class Scheduler(nn.Module):
    def __init__(self, base_scheduler):
        super().__init__()
        self.base_scheduler = base_scheduler

    def step(self, trans_log_variance, model_output, sample):
        factor = Adaptive(trans_log_variance)
        sigma = self.base_scheduler.sigmas[0]
        # 自适应噪声强度的实质是，把原始一步更新强度局部缩小
        adaptive_sigma = sigma*factor
        x_denoised = sample - adaptive_sigma * model_output
        return x_denoised

        

def Adaptive(trans_log_variance):
    scale = torch.exp(0.5*trans_log_variance)
    a = 4.0
    linear_part = (0.6 / a) * scale.abs() + 0.4
    flat_part = torch.ones_like(scale)
    # 标准差在0~4的时候，遵循自定义的线性变换，大于4的时候直接等于1
    factor = torch.where(scale.abs() < a, linear_part, flat_part)
    shift = 0.
    return factor
    

# 纯工具，从标准高斯里采样epsilon
def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional[Union[str, "torch.device"]] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    if isinstance(device, str):
        device = torch.device(device)
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                print(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slightly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents