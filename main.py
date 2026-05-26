import torch
import torch.nn as nn

import hydra
from omegaconf import DictConfig

from torchinfo import summary

from models.unet import UNet


def get_model_summary(cfg: DictConfig, model: nn.Module) -> None:
    input_size = (1, cfg.in_channels, cfg.image_size, cfg.image_size)
    summary(model,
            input_size=input_size,
            verbose=1,
            col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"),
            )

@hydra.main(version_base="1.3", config_path="./configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=== Config (resolved) ===")
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(device)
    model = UNet(cfg=cfg.model).to(device)
    # print(model)
    # Model Summary Torchinfo
    # get_model_summary(cfg, model)
    # 
    x = torch.randn([1, 3, 224, 224]).to(device)
    out = model(x)
    print(out.shape)  # This should print torch.Size([1, 1, 224, 224])


if __name__ == '__main__':
    main()
    