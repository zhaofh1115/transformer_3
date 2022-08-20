# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)
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

"""Multi-Head Attention layer definition."""

import math
from typing import Tuple

import torch
from torch import nn
from wenet.transformer.efficientconformer_attention import  (
    # Abs Attentions
    MultiHeadAttention,
    GroupedMultiHeadAttention,
    LocalMultiHeadAttention,
    StridedMultiHeadAttention,
    StridedLocalMultiHeadAttention,
    MultiHeadLinearAttention,
    # Rel Attentions
    RelPosMultiHeadSelfAttention,
    LocalRelPosMultiHeadSelfAttention,
    GroupedRelPosMultiHeadSelfAttention,
    StridedRelPosMultiHeadSelfAttention,
    StridedLocalRelPosMultiHeadSelfAttention
)


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """
    def __init__(self, n_head: int, n_feat: int, dropout_rate: float):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.n_feat = n_feat
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor, size
                (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(
        self, value: torch.Tensor, scores: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool)
    ) -> torch.Tensor:
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value, size
                (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score, size
                (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask, size (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        # NOTE(xcsong): When will `if mask.size(2) > 0` be True?
        #   1. onnx(16/4) [WHY? Because we feed real cache & real mask for the
        #           1st chunk to ease the onnx export.]
        #   2. pytorch training
        if mask.size(2) > 0 :  # time2 > 0
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            # For last chunk, time2 might be larger than scores.size(-1)
            mask = mask[:, :, :, :scores.size(-1)]  # (batch, 1, *, time2)
            scores = scores.masked_fill(mask, -float('inf'))
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0)  # (batch, head, time1, time2)
        # NOTE(xcsong): When will `if mask.size(2) > 0` be False?
        #   1. onnx(16/-1, -1/-1, 16/0)
        #   2. jit (16/-1, -1/-1, 16/0, 16/4)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).reshape(n_batch, -1,  self.n_feat)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
                pos_emb: torch.Tensor = torch.empty(0),
                cache: torch.Tensor = torch.zeros((0, 0, 0, 0))
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
                1.When applying cross attention between decoder and encoder,
                the batch padding mask for input is in (#batch, 1, T) shape.
                2.When applying self attention of encoder,
                the mask is in (#batch, T, T)  shape.
                3.When applying self attention of decoder,
                the mask is in (#batch, L, L)  shape.
                4.If the different position in decoder see different block
                of the encoder, such as Mocha, the passed in mask could be
                in (#batch, L, T) shape. But there is no such case in current
                Wenet.
            cache (torch.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`


        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
            torch.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`

        """
        q, k, v = self.forward_qkv(query, key, value)

        # NOTE(xcsong):
        #   when export onnx model, for 1st chunk, we feed
        #       cache(1, head, 0, d_k * 2) (16/-1, -1/-1, 16/0 mode)
        #       or cache(1, head, real_cache_t, d_k * 2) (16/4 mode).
        #       In all modes, `if cache.size(0) > 0` will alwayse be `True`
        #       and we will always do splitting and
        #       concatnation(this will simplify onnx export). Note that
        #       it's OK to concat & split zero-shaped tensors(see code below).
        #   when export jit  model, for 1st chunk, we always feed
        #       cache(0, 0, 0, 0) since jit supports dynamic if-branch.
        # >>> a = torch.ones((1, 2, 0, 4))
        # >>> b = torch.ones((1, 2, 3, 4))
        # >>> c = torch.cat((a, b), dim=2)
        # >>> torch.equal(b, c)        # True
        # >>> d = torch.split(a, 2, dim=-1)
        # >>> torch.equal(d[0], d[1])  # True
        if cache.size(0) > 0:
            key_cache, value_cache = torch.split(
                cache, cache.size(-1) // 2, dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        # NOTE(xcsong): We do cache slicing in encoder.forward_chunk, since it's
        #   non-trivial to calculate `next_cache_start` here.
        new_cache = torch.cat((k, v), dim=-1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask), new_cache


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """
    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k)) # 定义俩可训练的矩阵
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u) # 初始化这两个可训练的矩阵
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x, zero_triu: bool = False):
        """Compute relative positinal encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, size).
            zero_triu (bool): If true, return the lower triangular part of
                the matrix.
        Returns:
            torch.Tensor: Output tensor.
        """

        zero_pad = torch.zeros((x.size()[0], x.size()[1], x.size()[2], 1),
                               device=x.device,
                               dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(x.size()[0],
                                 x.size()[1],
                                 x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query: torch.Tensor,
                key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
                pos_emb: torch.Tensor = torch.empty(0),
                cache: torch.Tensor = torch.zeros((0, 0, 0, 0))
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, time2, size).
            cache (torch.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
            torch.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        # NOTE(xcsong):
        #   when export onnx model, for 1st chunk, we feed
        #       cache(1, head, 0, d_k * 2) (16/-1, -1/-1, 16/0 mode)
        #       or cache(1, head, real_cache_t, d_k * 2) (16/4 mode).
        #       In all modes, `if cache.size(0) > 0` will alwayse be `True`
        #       and we will always do splitting and
        #       concatnation(this will simplify onnx export). Note that
        #       it's OK to concat & split zero-shaped tensors(see code below).
        #   when export jit  model, for 1st chunk, we always feed
        #       cache(0, 0, 0, 0) since jit supports dynamic if-branch.
        # >>> a = torch.ones((1, 2, 0, 4))
        # >>> b = torch.ones((1, 2, 3, 4))
        # >>> c = torch.cat((a, b), dim=2)
        # >>> torch.equal(b, c)        # True
        # >>> d = torch.split(a, 2, dim=-1)
        # >>> torch.equal(d[0], d[1])  # True
        if cache.size(0) > 0:
            key_cache, value_cache = torch.split(
                cache, cache.size(-1) // 2, dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        # NOTE(xcsong): We do cache slicing in encoder.forward_chunk, since it's
        #   non-trivial to calculate `next_cache_start` here.
        new_cache = torch.cat((k, v), dim=-1)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        # Remove rel_shift since it is useless in speech recognition,
        # and it requires special attention for streaming.
        # matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k)  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask), new_cache



# class GroupedRelPosMultiHeadSelfAttention(MultiHeadedAttention):
#     """Multi-Head Attention layer with relative position encoding.
#     Paper: https://arxiv.org/abs/1901.02860
#     Args:
#         n_head (int): The number of heads.
#         n_feat (int): The number of features.
#         dropout_rate (float): Dropout rate.
#     """
#     def __init__(self, n_head, n_feat, dropout_rate, group_size):
#         """Construct an RelPositionMultiHeadedAttention object."""
#         super().__init__(n_head, n_feat, dropout_rate)
#         # linear transformation for positional encoding
#         self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
#         # these two learnable bias are used in matrix c and matrix d
#         # as described in https://arxiv.org/abs/1901.02860 Section 3.3
#         self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k)) # 定义俩可训练的矩阵
#         self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
#         torch.nn.init.xavier_uniform_(self.pos_bias_u) # 初始化这两个可训练的矩阵
#         torch.nn.init.xavier_uniform_(self.pos_bias_v)
#         self.group_size = group_size # G
#         self.dim_head = (self.group_size * n_feat) // self.n_head # d
#         self.n_feat = n_feat
        

#     def rel_shift(self, x, zero_triu: bool = False):
#         """Compute relative positinal encoding.
#         Args:
#             x (torch.Tensor): Input tensor (batch, time, size).
#             zero_triu (bool): If true, return the lower triangular part of
#                 the matrix.
#         Returns:
#             torch.Tensor: Output tensor.
#         """

#         zero_pad = torch.zeros((x.size()[0], x.size()[1], x.size()[2], 1),
#                                device=x.device,
#                                dtype=x.dtype)
#         x_padded = torch.cat([zero_pad, x], dim=-1)

#         x_padded = x_padded.view(x.size()[0],
#                                  x.size()[1],
#                                  x.size(3) + 1, x.size(2))
#         x = x_padded[:, :, 1:].view_as(x)

#         if zero_triu:
#             ones = torch.ones((x.size(2), x.size(3)))
#             x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

#         return x

#     def forward(self, query: torch.Tensor,
#                 key: torch.Tensor, value: torch.Tensor,
#                 mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
#                 pos_emb: torch.Tensor = torch.empty(0),
#                 cache: torch.Tensor = torch.zeros((0, 0, 0, 0))
#                 ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
#         Args:
#             query (torch.Tensor): Query tensor (#batch, time1, size).
#             key (torch.Tensor): Key tensor (#batch, time2, size).
#             value (torch.Tensor): Value tensor (#batch, time2, size).
#             mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
#                 (#batch, time1, time2), (0, 0, 0) means fake mask.
#             pos_emb (torch.Tensor): Positional embedding tensor
#                 (#batch, time2, size).
#             cache (torch.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
#                 where `cache_t == chunk_size * num_decoding_left_chunks`
#                 and `head * d_k == size`
#         Returns:
#             torch.Tensor: Output tensor (#batch, time1, d_model).
#             torch.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
#                 where `cache_t == chunk_size * num_decoding_left_chunks`
#                 and `head * d_k == size`
#         """
#         batch_size = query.size(0)
#         q, k, v = self.forward_qkv(query, key, value)
#         q = q.transpose(1, 2)  # (batch, time1, head, d_k)


#         if cache.size(0) > 0:
#             key_cache, value_cache = torch.split(
#                 cache, cache.size(-1) // 2, dim=-1)
#             k = torch.cat([key_cache, k], dim=2)
#             v = torch.cat([value_cache, v], dim=2)
#         # NOTE(xcsong): We do cache slicing in encoder.forward_chunk, since it's
#         #   non-trivial to calculate `next_cache_start` here.
#         new_cache = torch.cat((k, v), dim=-1)

#         n_batch_pos = pos_emb.size(0)
#         p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
#         # p = p.transpose(1, 2)  # (batch, head, time1, d_k)

#         # (batch, head, time1, d_k)
#         q_with_bias_u = q + self.pos_bias_u
#         # (batch, head, time1, d_k)
#         q_with_bias_v = q + self.pos_bias_v

#         # reshape efficientconformer
#         Qu = q_with_bias_u.reshape(batch_size, -1, self.n_head, self.dim_head).transpose(1, 2)
#         Qv = q_with_bias_v.reshape(batch_size, -1, self.n_head, self.dim_head).transpose(1, 2)

#         k = k.transpose(1, 2)
#         v = v.transpose(1, 2)
#         k = k.reshape(batch_size, -1, self.n_head, self.dim_head).transpose(1, 2)
#         v = v.reshape(batch_size, -1, self.n_head, self.dim_head).transpose(1, 2)

#         p = p.reshape(batch_size, -1, self.n_head, self.dim_head).transpose(1, 2)

#         matrix_ac = torch.matmul(Qu, k.transpose(-2, -1))
#         matrix_bd = torch.matmul(Qv, p.transpose(-2, -1))
#         att_scores = (matrix_ac + matrix_bd) / math.sqrt(
#             self.dim_head)  # (batch, head, time1, time2)

#         # att_scores_K = Qu.matmul(k.transpose(2, 3))
#         # att_scores_p = self.rel_to_abs(Qv.matmul(p.transpose(2, 3)))
#         # att_scores = (att_scores_K + att_scores_p) / k.shape[-1]**0.5

#         mask = mask[:, :, ::self.group_size, ::self.group_size]


#         return self.forward_attention(v, att_scores, mask), new_cache



class MultiHeadSelfAttentionModule(nn.Module):

    """Multi-Head Self-Attention Module

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        Pdrop: residual dropout probability
        max_pos_encoding: maximum position
        relative_pos_enc: whether to use relative postion embedding
        causal: True for causal attention with masked future context
        group_size: Attention group size
        kernel_size: Attention kernel size
        stride: Query stride
        linear_att: whether to use multi-head linear self-attention

    """

    def __init__(self, dim_model, num_heads, Pdrop, max_pos_encoding, relative_pos_enc, causal, group_size, kernel_size, stride, linear_att):
        super(MultiHeadSelfAttentionModule, self).__init__()

        # kernel_size = 15, group_size = [3, 1, 1], stride = 1

        # Assert
        assert not (group_size > 1 and kernel_size is not None), "Local grouped attention not implemented"
        assert not (group_size > 1 and stride > 1 is not None), "Strided grouped attention not implemented"
        assert not (linear_att and relative_pos_enc), "Linear attention requires absolute positional encodings"

        # Pre Norm
        self.norm = nn.LayerNorm(dim_model, eps=1e-6)

        # Multi-Head Linear Attention
        if linear_att:
            self.mhsa = MultiHeadLinearAttention(dim_model, num_heads)

        # Grouped Multi-Head Self-Attention
        elif group_size > 1:
            if relative_pos_enc:
                self.mhsa = GroupedRelPosMultiHeadSelfAttention(dim_model, num_heads, causal, max_pos_encoding, group_size)
            else:
                self.mhsa = GroupedMultiHeadAttention(dim_model, num_heads, group_size)
        
        # Local Multi-Head Self-Attention
        elif kernel_size is not None and stride == 1:
            if relative_pos_enc:
                self.mhsa = LocalRelPosMultiHeadSelfAttention(dim_model, num_heads, causal, kernel_size)
            else:
                self.mhsa = LocalMultiHeadAttention(dim_model, num_heads, kernel_size)

        # Strided Multi-Head Self-Attention
        elif kernel_size is None and stride > 1:
            if relative_pos_enc:
                self.mhsa = StridedRelPosMultiHeadSelfAttention(dim_model, num_heads, causal, max_pos_encoding, stride)
            else:
                self.mhsa = StridedMultiHeadAttention(dim_model, num_heads, stride)

        # Strided Local Multi-Head Self-Attention
        elif stride > 1 and kernel_size is not None:
            if relative_pos_enc:
                self.mhsa = StridedLocalRelPosMultiHeadSelfAttention(dim_model, num_heads, causal, kernel_size, stride)
            else:
                self.mhsa = StridedLocalMultiHeadAttention(dim_model, num_heads, kernel_size, stride)

        # Multi-Head Self-Attention
        else:
            if relative_pos_enc:
                self.mhsa = RelPosMultiHeadSelfAttention(dim_model, num_heads, causal, max_pos_encoding)
            else:
                self.mhsa = MultiHeadAttention(dim_model, num_heads)
            
        # Dropout
        self.dropout = nn.Dropout(Pdrop)

        # Module Params
        self.rel_pos_enc = relative_pos_enc
        self.linear_att = linear_att

    def forward(self, x, mask: torch.Tensor, att_cache: torch.tensor):

        # Pre Norm
        x = self.norm(x)

        # Multi-Head Self-Attention
        if self.linear_att:
            x, attention = self.mhsa(x, x, x)
        elif self.rel_pos_enc:
            x, mask, new_cache = self.mhsa(x, x, x, mask, att_cache)
        # else:
        #     x, attention = self.mhsa(x, x, x, mask)

        # Dropout
        x = self.dropout(x)

        return x, mask, new_cache
