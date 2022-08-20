# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
#               2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)

"""Encoder self-attention layer definition."""

from typing import Optional, Tuple

import torch
from torch import nn

from wenet.transformer.positionwise_feed_forward import FeedForwardModule
from wenet.transformer.attention import MultiHeadSelfAttentionModule
from wenet.transformer.convolution import ConvolutionModuleEfficient

from wenet.transformer.efficientconvlayers import (
    Linear,
    Conv1d,
    Transpose,
    DepthwiseSeparableConv1d,
    Swish,
    Glu
)

class TransformerEncoderLayer(nn.Module):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: to use layer_norm after each sub-block.
        concat_after (bool): Whether to concat attention layer's input and
            output.
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)

    """
    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward: torch.nn.Module,
        dropout_rate: float,
        normalize_before: bool = True,
        concat_after: bool = False,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(size, eps=1e-5)
        self.norm2 = nn.LayerNorm(size, eps=1e-5)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        else:
            self.concat_linear = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (torch.Tensor): (#batch, time, size)
            mask (torch.Tensor): Mask tensor for the input (#batch, time，time),
                (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): just for interface compatibility
                to ConformerEncoderLayer
            mask_pad (torch.Tensor): does not used in transformer layer,
                just for unified api with conformer.
            att_cache (torch.Tensor): Cache tensor of the KEY & VALUE
                (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
            cnn_cache (torch.Tensor): Convolution cache in conformer layer
                (#batch=1, size, cache_t2), not used here, it's for interface
                compatibility to ConformerEncoderLayer.
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time, time).
            torch.Tensor: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
            torch.Tensor: cnn_cahce tensor (#batch=1, size, cache_t2).

        """
        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        x_att, new_att_cache = self.self_attn(
            x, x, x, mask, cache=att_cache)
        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        fake_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        return x, mask, new_att_cache, fake_cnn_cache


class ConformerEncoderLayer(nn.Module):
    """Encoder layer module.
    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module
             instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: use layer_norm after each sub-block.
        concat_after (bool): Whether to concat attention layer's input and
            output.
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
    """
    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward: Optional[nn.Module] = None,
        feed_forward_macaron: Optional[nn.Module] = None,
        conv_module: Optional[nn.Module] = None,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
        concat_after: bool = False,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = nn.LayerNorm(size, eps=1e-5)  # for the FNN module
        self.norm_mha = nn.LayerNorm(size, eps=1e-5)  # for the MHA module
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-5)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = nn.LayerNorm(size,
                                          eps=1e-5)  # for the CNN module
            self.norm_final = nn.LayerNorm(
                size, eps=1e-5)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        else:
            self.concat_linear = nn.Identity()


    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (torch.Tensor): (#batch, time, size)
            mask (torch.Tensor): Mask tensor for the input (#batch, time，time),
                (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): positional encoding, must not be None
                for ConformerEncoderLayer.
            mask_pad (torch.Tensor): batch padding mask used for conv module.
                (#batch, 1，time), (0, 0, 0) means fake mask.
            att_cache (torch.Tensor): Cache tensor of the KEY & VALUE
                (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
            cnn_cache (torch.Tensor): Convolution cache in conformer layer
                (#batch=1, size, cache_t2)
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time, time).
            torch.Tensor: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
            torch.Tensor: cnn_cahce tensor (#batch, size, cache_t2).
        """

        # whether to use macaron style(laynorm-linear-activate-dropout-linear-dropout-0.5-+x) 
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(
                self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # multi-headed self-attention module
        # mhsa:x+Dropout(Laynorm(laynorm-qkvLinear-qkvtranspose-scores-mask-softmaxscoresmask-dropoutsoftmaxscoresmask-x(dropoutsoftmaxscoresmask*v)-linear))
        # relpos_mhsa:(laynorm-qkvLinear-qkvtranspose-qtranspose-pos_emblinear(p)-ptranspose-q+biastranspose-q+bias v+bias-scores--mask-softmaxscoresmask-dropoutsoftmaxscoresmask-x(dropoutsoftmaxscoresmask*v)-linear)
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        x_att, new_att_cache = self.self_attn(
            x, x, x, mask, pos_emb, att_cache)
        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # convolution module
        # Fake new cnn cache here, and then change it in conv_module
        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)

            if not self.normalize_before:
                x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)

        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        return x, mask, new_att_cache, new_cnn_cache

class EfficientConformerEncoderLayer(nn.Module):
    def __init__(
        self, 
        dim_model, 
        dim_expand, 
        ff_ratio, 
        num_heads, 
        kernel_size, 
        att_group_size, 
        att_kernel_size,
        linear_att,
        Pdrop, 
        relative_pos_enc, 
        max_pos_encoding, 
        conv_stride,
        att_stride,
        causal,
    ):
        super(EfficientConformerEncoderLayer, self).__init__()

        # Feed Forward Module 1
        self.feed_forward_module1 = FeedForwardModule(
            dim_model=dim_model, 
            dim_ffn=dim_model * ff_ratio,
            Pdrop=Pdrop, 
            act="swish",
            inner_dropout=True
        )

        # Multi-Head Self-Attention Module
        self.multi_head_self_attention_module = MultiHeadSelfAttentionModule(
            dim_model=dim_model, 
            num_heads=num_heads,  
            Pdrop=Pdrop, 
            max_pos_encoding=max_pos_encoding,
            relative_pos_enc=relative_pos_enc, 
            causal=causal,
            group_size=att_group_size,
            kernel_size=att_kernel_size,
            stride=att_stride,
            linear_att=linear_att,
        )

        # Convolution Module
        self.convolution_module = ConvolutionModuleEfficient(
            dim_model=dim_model,
            dim_expand=dim_expand,
            kernel_size=kernel_size, 
            Pdrop=Pdrop, 
            stride=conv_stride,
            padding="causal" if causal else "same",
        )

        # Feed Forward Module 2
        self.feed_forward_module2 = FeedForwardModule(
            dim_model=dim_expand, 
            dim_ffn=dim_expand * ff_ratio,
            Pdrop=Pdrop, 
            act="swish",
            inner_dropout=True
        )

        # Block Norm
        self.norm = nn.LayerNorm(dim_expand, eps=1e-5)

        # Attention Residual
        self.att_res = nn.Sequential(
            Transpose(1, 2),
            nn.MaxPool1d(kernel_size=1, stride=att_stride),
            Transpose(1, 2)
        ) if att_stride > 1 else nn.Identity()

        # Convolution Residual
        self.conv_res = nn.Sequential(
            Transpose(1, 2),
            Conv1d(dim_model, dim_expand, kernel_size=1, stride=conv_stride),
            Transpose(1, 2)
        ) if dim_model != dim_expand else nn.Sequential(
            Transpose(1, 2),
            nn.MaxPool1d(kernel_size=1, stride=conv_stride),
            Transpose(1, 2)
        ) if conv_stride > 1 else nn.Identity()

        # Bloc Stride
        self.stride = conv_stride * att_stride

    def forward(
        self, 
        x, 
        mask: torch.Tensor, 
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0))):

        # FFN Module 1(laynorm-linear-activate-dropout-linear-dropout-0.5-+x)
        x = x + 1/2 * self.feed_forward_module1(x)

        # MHSA Module
        # (laynorm-qkvLinear-qkvtranspose-qtranspose-pos_emblinear(p)-ptranspose-q+biastranspose-q+bias v+bias-scores--mask-softmaxscoresmask-dropoutsoftmaxscoresmask-x(dropoutsoftmaxscoresmask*v)-linear)
        # laynorm-qkvLinear-KhVhKV-qkvpad-q+bias v+bias-QKpos-linear-q+bias v+bias(QKpos-linear)transpose
        x_att, mask, new_att_cache = self.multi_head_self_attention_module(x, mask, att_cache)
        x = self.att_res(x) + x_att

        # Conv Module
        x, new_cnn_cache = self.convolution_module(x, cnn_cache)
        x = self.conv_res(x) + self.convolution_module(x)

        # FFN Module 2
        x = x + 1/2 * self.feed_forward_module2(x)

        # Block Norm
        x = self.norm(x)

        return x, mask, new_att_cache, new_cnn_cache
