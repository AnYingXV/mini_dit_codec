````markdown
# Self DiT-IC Reproduction

Original code repository: https://github.com/Eric-qi/DiT-IC

Arxiv: https://arxiv.org/abs/2603.13162

## Download the pretrained DiT model

This reproduction uses the pretrained SANA model in Diffusers format:

```text
Efficient-Large-Model/Sana_600M_1024px_diffusers
````

First, install or update `huggingface_hub`:

```bash
pip install -U huggingface_hub
```

Then download the pretrained DiT model from Hugging Face:

```bash
hf download Efficient-Large-Model/Sana_600M_1024px_diffusers \
  --local-dir ./SANA
```

```

The local DiT path can then be set to:

```python
dit_path = "./SANA"
```

For slow or unstable network connections, increase the Hugging Face download timeout:

```bash
HF_HUB_DOWNLOAD_TIMEOUT=120 \
hf download Efficient-Large-Model/Sana_600M_1024px_diffusers \
  --local-dir ./SANA
```

To download only the components required by the current DiT-IC implementation:

```bash
HF_HUB_DOWNLOAD_TIMEOUT=120 \
hf download Efficient-Large-Model/Sana_600M_1024px_diffusers \
  --local-dir ./SANA \
  --include "scheduler/*" \
  --include "transformer/*" \
  --include "vae/*" \
  --include "model_index.json"
```

```
```
