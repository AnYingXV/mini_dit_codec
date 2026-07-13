import torch
from model.LatentCodec import latent_codec

codec = latent_codec()

mask_0, mask_1, mask_2, mask_3 = codec.get_mask(
    1, 4, 2, 2, device="cpu"
)
sth = torch.tensor([[
    [[1,2],
    [3,4]],

    [[5,6], 
    [7,8]],

    [[9,10],
    [11,12]],

    [[13,14],
    [15,16]]
    ]])

squeezed = codec.squeeze_with_mask(sth, mask_0)
print(squeezed)
unsqueezed = codec.unsqueeze_with_mask(squeezed, mask_0)
print(unsqueezed)
