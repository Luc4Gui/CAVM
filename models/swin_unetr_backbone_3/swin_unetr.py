"""SwinUNETR with cross attention."""
from typing import MutableMapping, Sequence, Tuple, Union

import torch

from monai.networks import blocks
from monai.networks.nets import swin_unetr

__all__ = ["SwinUNETR"]

FeaturesDictType = MutableMapping[str, torch.Tensor]


class SwinUNETR(swin_unetr.SwinUNETR):
    """SwinUNETR with cross attention."""

    def __init__(
        self,
        img_size: Union[Sequence[int], int],
        *args,
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: Union[Tuple, str] = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        spatial_dims: int = 3,
        **kwargs,
    ) -> None:
        """
        """
        super().__init__(
            img_size,
            *args,
            num_heads=num_heads,
            feature_size=feature_size,
            norm_name=norm_name,
            spatial_dims=spatial_dims,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            **kwargs,
        )

        self.swinViT_0 = swin_unetr.SwinTransformer(
            in_chans=4,
            embed_dim=feature_size,
            window_size=self.swinViT.window_size,
            patch_size=self.swinViT.patch_size,
            depths=(2,2,2,2),
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=0,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=False,
            spatial_dims=spatial_dims,
            downsample="merging",
            use_v2=False,
        )

        self.swinViT_0_0 = swin_unetr.SwinTransformer(
            in_chans=4,
            embed_dim=feature_size,
            window_size=self.swinViT.window_size,
            patch_size=self.swinViT.patch_size,
            depths=(2,2,2,2),
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=0,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=False,
            spatial_dims=spatial_dims,
            downsample="merging",
            use_v2=False,
        )

        self.encoder1 = blocks.UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoderT0 = blocks.UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoderT1 = blocks.UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder11 = blocks.UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
    
    def forward_encoder(self, x):
        x_hiddens = self.swinViT(x, self.normalize)
        xs_dec4 = self.encoder10(x_hiddens[4])
        xs_dec3 = self.encoder11(x_hiddens[3])
        return xs_dec3, xs_dec4
    
    def forward_encoder_0(self, x):
        """Encode features."""
        x_hiddens = self.swinViT_0(x, self.normalize)
        x_enc0 = self.encoder1(x)
        x_enc1 = self.encoder2(x_hiddens[0])
        x_enc2 = self.encoder3(x_hiddens[1])
        x_enc3 = self.encoder4(x_hiddens[2])
        return {"enc0": x_enc0, "enc1": x_enc1, "enc2": x_enc2, "enc3": x_enc3}

    def forward_encoder_0_0(self, x):
        """Encode features."""
        x_hiddens = self.swinViT_0_0(x, self.normalize)
        x_enc4 = self.encoderT1(x_hiddens[3])
        x_dec4 = self.encoderT0(x_hiddens[4])
        return x_enc4, x_dec4

    def forward_decoder(self, x_encoded3, x_encoded4, x0_encoded) -> torch.Tensor:
        """Decode features."""
        x_dec3 = self.decoder5(x_encoded4, x_encoded3)
        x_dec2 = self.decoder4(x_dec3, x0_encoded["enc3"])
        x_dec1 = self.decoder3(x_dec2, x0_encoded["enc2"])
        x_dec0 = self.decoder2(x_dec1, x0_encoded["enc1"])
        x_out = self.decoder1(x_dec0, x0_encoded["enc0"])
        xs_logits = self.out(x_out)
        return xs_logits

    def forward(self, x, x0) -> Sequence[torch.Tensor]:
        """Two views forward."""
        x_encoded3, x_encoded4 = self.forward_encoder(x)
        x0_encoded = self.forward_encoder_0(x0)
        return self.forward_decoder(x_encoded3, x_encoded4, x0_encoded)
    
    def forward_0(self, x, x0) -> Sequence[torch.Tensor]:
        """Two views forward."""
        x_encoded3, x_encoded4 = self.forward_encoder_0_0(x0)
        x0_encoded = self.forward_encoder_0(x0)
        return self.forward_decoder(x_encoded3, x_encoded4, x0_encoded)
    
    def no_weight_decay(self):
        """Disable weight_decay on specific weights."""
        nwd = {"swinViT.absolute_pos_embed"}
        for n, _ in self.named_parameters():
            if "relative_position_bias_table" in n:
                nwd.add(n)
        return nwd

    def group_matcher(self, coarse=False):
        """Layer counting helper, used by timm."""
        return dict(
            stem=r"^swinViT\.absolute_pos_embed|patch_embed",  # stem and embed
            blocks=r"^swinViT\.layers(\d+)\.0"
            if coarse
            else [
                (r"^swinViT\.layers(\d+)\.0.downsample", (0,)),
                (r"^swinViT\.layers(\d+)\.0\.\w+\.(\d+)", None),
                (r"^swinViT\.norm", (99999,)),
            ],
        )
    