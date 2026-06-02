import torch
import torch.nn as nn

import hydra
from omegaconf import DictConfig

from torchinfo import summary

from models.resvss_unet import ResVSSUNet


def get_model_summary(cfg: DictConfig, model: nn.Module) -> None:
    input_size = (1, cfg.in_channels, cfg.image_size, cfg.image_size)
    print("\n=== Model summary ===")
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
    model = ResVSSUNet(cfg=cfg.model).to(device)

    # Load pretrained weights
    if cfg.use_pretrained_ckpt:
        model.load_from(cfg.paths.pretrained_ckpt)

    # print(model)
    # Model Summary Torchinfo
    get_model_summary(cfg, model)
    # Test Output
    print("\n=== Test Output ===")
    input = torch.randn([1, cfg.in_channels, cfg.image_size, cfg.image_size]).to(device)
    output = model(input)
    print(output.shape)  # Expected: (B, num_classes, H, W)


if __name__ == '__main__':
    main()
    