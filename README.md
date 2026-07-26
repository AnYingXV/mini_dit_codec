# Self DiT-IC Reproduction

Original code repository: https://github.com/Eric-qi/DiT-IC

Arxiv: https://arxiv.org/abs/2603.13162

## Download the pretrained DiT model

This reproduction uses the pretrained SANA model in Diffusers format:

```text
Efficient-Large-Model/Sana_600M_1024px_diffusers
```

First, install or update `huggingface_hub`:

```bash
pip install -U huggingface_hub
```

Set the Hugging Face mirror endpoint:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Then download the pretrained DiT model:

```bash
hf download Efficient-Large-Model/Sana_600M_1024px_diffusers \
  --local-dir ./SANA
```

The local DiT path can then be set to:

```python
dit_path = "./SANA"
```
