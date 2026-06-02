import math
import torch
import torch.nn as nn

from torchvision.models import resnet34

from timm.layers import DropPath, trunc_normal_
from models.vmamba.patch import PatchEmbed2D, PatchExpand2D, Final_PatchExpand2D
from models.vmamba.vsslayers import VSSLayer_up

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

class ResNetVMUNet(nn.Module):
    def __init__(self, num_classes=1000, patch_size=2, enc_channels=[64, 64, 128, 256, 512], dims=[96, 192, 384, 768],
                 depths_decoder=[2, 2, 2, 2], dims_decoder=[768, 384, 192, 96], d_state=16, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_layers = len(depths_decoder)
        resnet = resnet34(weights='DEFAULT')
        self.enc_layer = nn.ModuleList()
        self.enc_layer.append(nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu))
        self.enc_layer.append(nn.Sequential(resnet.maxpool, resnet.layer1))
        self.enc_layer.append(resnet.layer2)
        self.enc_layer.append(resnet.layer3)

        self.proj = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.proj.append(PatchEmbed2D(in_chans=enc_channels[i_layer], embed_dim=dims[i_layer],
                                          patch_size=patch_size, norm_layer=norm_layer))

        self.last_layer = nn.Sequential(resnet.layer4,
                                        PatchEmbed2D(in_chans=enc_channels[-1], embed_dim=dims[-1],
                                                     patch_size=1, norm_layer=norm_layer))

        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]
        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer_up(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                d_state=math.ceil(dims_decoder[0] / 6) if d_state is None else d_state,  # 20240109
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None,
            )
            self.layers_up.append(layer)

        self.final_up = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)
        self.final_conv = nn.Conv2d(dims_decoder[-1] // 4, num_classes, 1)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless

        Conv2D is not intialized !!!
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        skips_list = []
        for inx, proj_layer in enumerate(self.proj):
            x = self.enc_layer[inx](x)
            skip = proj_layer(x)
            x = self.pos_drop(x)
            skips_list.append(skip)
        x = self.last_layer(x)
        return self.pos_drop(x), skips_list

    def forward_features_up(self, x, skip_list):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = layer_up(x + skip_list[-inx])
        return x

    def forward_final(self, x):
        x = self.final_up(x)
        x = x.permute(0, 3, 1, 2)
        x = self.final_conv(x)
        return x

    def forward(self, x):
        x, skip_list = self.forward_features(x)
        x = self.forward_features_up(x, skip_list)
        x = self.forward_final(x)
        return x