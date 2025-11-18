import torch
import torch.nn as nn

# Correct import because SUNet_detail.py is in the same package
from .SUNet_detail import SUNet


class SUNet_model(nn.Module):
    def __init__(self, config):
        super(SUNet_model, self).__init__()
        self.config = config

        self.swin_unet = SUNet(
            img_size=config['SWINUNET']['IMG_SIZE'],
            patch_size=config['SWINUNET']['PATCH_SIZE'],
            in_chans=3,
            out_chans=3,
            embed_dim=config['SWINUNET']['EMB_DIM'],
            depths=config['SWINUNET']['DEPTH_EN'],
            num_heads=config['SWINUNET']['HEAD_NUM'],
            window_size=config['SWINUNET']['WIN_SIZE'],
            mlp_ratio=config['SWINUNET']['MLP_RATIO'],
            qkv_bias=config['SWINUNET']['QKV_BIAS'],
            qk_scale=config['SWINUNET']['QK_SCALE'],
            drop_rate=config['SWINUNET']['DROP_RATE'],
            drop_path_rate=config['SWINUNET']['DROP_PATH_RATE'],
            ape=config['SWINUNET']['APE'],
            patch_norm=config['SWINUNET']['PATCH_NORM'],
            use_checkpoint=config['SWINUNET']['USE_CHECKPOINTS']
        )

    def forward(self, x):
        # convert grayscale to pseudo-RGB
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.swin_unet(x)


if __name__ == '__main__':
    import os
    import yaml
    from thop import profile
    from ..utils.model_utils import network_parameters

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'training.yaml')
    config_path = os.path.abspath(config_path)

    with open(config_path, 'r') as f:
        opt = yaml.safe_load(f)

    model = SUNet_model(opt)

    # test input: 3 channels, not 156 (that was wrong)
    x = torch.randn(1, 3, 256, 256)

    out = model(x)
    flops, params = profile(model, (x,))

    print(out.shape)
    print(f"FLOPs: {flops}")
    print(f"Params: {params}")
