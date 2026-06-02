from omegaconf import DictConfig

import torch
from torch.nn import Module, LayerNorm

from models.hybrid_model.resnet_vmunet import ResNetVMUNet

class ResVSSUNet(Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.hybridvmunet = ResNetVMUNet(num_classes=cfg.out_channels, patch_size=cfg.patch_size,
                                         enc_channels=cfg.enc_channels, dims=cfg.dims,
                                         depths_decoder=cfg.depths_decoder, dims_decoder=cfg.dims_decoder,
                                         drop_path_rate=cfg.drop_path_rate, norm_layer=LayerNorm)

    def forward(self, x):
        x = self.hybridvmunet(x)
        return torch.sigmoid(x)

    def load_from(self, load_ckpt_path):
        if load_ckpt_path is not None:
            print('=== Loading pretrained model from {} ==='.format(load_ckpt_path))
            model_dict = self.hybridvmunet.state_dict()
            modelcheckpoint = torch.load(load_ckpt_path)
            pretrained_odict = modelcheckpoint['model']
            pretrained_dict = {}
            for k, v in pretrained_odict.items():
                if 'layers.0' in k:
                    new_k = k.replace('layers.0', 'layers_up.3')
                    pretrained_dict[new_k] = v
                elif 'layers.1' in k:
                    new_k = k.replace('layers.1', 'layers_up.2')
                    pretrained_dict[new_k] = v
                elif 'layers.2' in k:
                    new_k = k.replace('layers.2', 'layers_up.1')
                    pretrained_dict[new_k] = v
                elif 'layers.3' in k:
                    new_k = k.replace('layers.3', 'layers_up.0')
                    pretrained_dict[new_k] = v
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict),
                                                                                       len(pretrained_dict),
                                                                                       len(new_dict)))
            self.hybridvmunet.load_state_dict(model_dict)

            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("decoder loaded finished!")