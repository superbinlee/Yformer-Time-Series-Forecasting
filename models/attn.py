from math import sqrt
import numpy as np
import torch
import torch.nn as nn

from utils.masking import TriangularCausalMask, ProbMask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        """
        全注意力（FullAttention）机制。

        参数：
        - mask_flag: 是否启用注意力掩码（用于因果推断）。
        - factor: 采样因子（未使用，兼容其他注意力机制）。
        - scale: 缩放因子（默认 1 / sqrt(E)）。
        - attention_dropout: 注意力权重的 Dropout 比例。
        - output_attention: 是否输出注意力权重。
        """
        super(FullAttention, self).__init__()
        self.scale = scale  # 注意力得分的缩放因子
        self.mask_flag = mask_flag  # 是否使用掩码
        self.output_attention = output_attention  # 是否输出注意力权重
        self.dropout = nn.Dropout(attention_dropout)  # Dropout 层

    def forward(self, queries, keys, values, attn_mask):
        """
        前向传播函数。

        参数：
        - queries: 查询向量，形状为 (B, L, H, E)，其中：
          - B: 批量大小。
          - L: 查询序列长度。
          - H: 多头注意力的头数。
          - E: 每个头的特征维度。
        - keys: 键向量，形状为 (B, S, H, E)，其中：
          - S: 键序列长度。
        - values: 值向量，形状为 (B, S, H, D)，其中：
          - D: 值的特征维度。
        - attn_mask: 注意力掩码，用于屏蔽特定位置的注意力。

        返回：
        - V: 注意力机制的输出，形状为 (B, L, H, D)。
        - A（可选）: 注意力权重矩阵，形状为 (B, H, L, S)（如果 output_attention=True）。
        """
        # 获取输入的形状
        B, L, H, E = queries.shape  # 批量大小、查询长度、多头数量、特征维度
        _, S, _, D = values.shape  # 键序列长度和值的特征维度
        # 如果未指定缩放因子，则使用默认值 1 / sqrt(E)
        scale = self.scale or 1. / sqrt(E)

        # 计算注意力得分（内积操作）
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)  # 形状为 (B, H, L, S)

        # 如果启用了掩码
        if self.mask_flag:
            if attn_mask is None:
                # 如果没有提供掩码，则生成一个三角因果掩码
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            # 将掩码位置的得分填充为负无穷（防止这些位置被注意力关注）
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # 对得分矩阵进行缩放和 Softmax 归一化，然后应用 Dropout
        A = self.dropout(torch.softmax(scale * scores, dim=-1))  # 注意力权重矩阵

        # 使用注意力权重矩阵与值向量进行加权求和，得到输出
        V = torch.einsum("bhls,bshd->blhd", A, values)  # 输出形状为 (B, L, H, D)

        # 如果启用输出注意力权重，则返回输出和权重矩阵
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            # 否则返回输出和 None（不输出注意力权重）
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        """
        稀疏注意力机制（ProbSparse Attention）。

        参数：
        - mask_flag: 是否启用注意力掩码（用于因果推断）。
        - factor: 稀疏注意力采样因子，用于控制计算复杂度。
        - scale: 缩放因子（默认 1 / sqrt(D)）。
        - attention_dropout: 注意力权重的 Dropout 比例。
        - output_attention: 是否输出注意力权重。
        """
        super(ProbAttention, self).__init__()
        self.factor = factor  # 控制稀疏计算复杂度
        self.scale = scale  # 注意力得分的缩放因子
        self.mask_flag = mask_flag  # 是否使用掩码
        self.output_attention = output_attention  # 是否输出注意力权重
        self.dropout = nn.Dropout(attention_dropout)  # Dropout 层

    def _prob_QK(self, Q, K, sample_k, n_top):
        """
        使用随机采样的方法计算注意力得分。

        参数：
        - Q: 查询向量，形状为 (B, H, L_Q, D)。
        - K: 键向量，形状为 (B, H, L_K, D)。
        - sample_k: 键的采样数量，用于减少计算开销。
        - n_top: 选择的前 n_top 个查询。

        返回：
        - Q_K: 稀疏查询和键的注意力得分矩阵。
        - M_top: 稀疏选择的查询索引。
        """
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # 键的随机采样
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # 从键中随机采样
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # 稀疏选择：计算每个查询的稀疏性度量
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]  # 选择稀疏性最高的 n_top 个查询

        # 使用选择的查询计算完整的注意力得分
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # 计算稀疏注意力得分

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        """
        初始化上下文向量。

        参数：
        - V: 值向量，形状为 (B, H, L_V, D)。
        - L_Q: 查询序列的长度。

        返回：
        - context: 初始化的上下文向量。
        """
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # 如果没有掩码，则直接对值向量进行平均（全局注意力）
            V_sum = V.mean(dim=-2)
            context = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            # 如果有掩码，计算前缀和（因果注意力）
            assert L_Q == L_V  # 自注意力时要求查询长度等于值长度
            context = V.cumsum(dim=-2)
        return context

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        """
        更新上下文向量。

        参数：
        - context_in: 初始化的上下文向量。
        - V: 值向量，形状为 (B, H, L_V, D)。
        - scores: 稀疏的注意力得分矩阵。
        - index: 选择的查询索引。
        - L_Q: 查询序列的长度。
        - attn_mask: 注意力掩码。

        返回：
        - context_in: 更新后的上下文向量。
        - attn: 注意力权重矩阵（可选）。
        """
        B, H, L_V, D = V.shape

        if self.mask_flag:
            # 如果使用掩码，构造稀疏掩码
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # 对稀疏得分矩阵进行 softmax 归一化
        attn = torch.softmax(scores, dim=-1)

        # 更新上下文向量
        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)

        if self.output_attention:
            # 如果输出注意力权重，构造完整的注意力权重矩阵
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        """
        前向传播函数。

        参数：
        - queries: 查询向量，形状为 (B, L_Q, H, D)。
        - keys: 键向量，形状为 (B, L_K, H, D)。
        - values: 值向量，形状为 (B, L_K, H, D)。
        - attn_mask: 注意力掩码。

        返回：
        - context: 注意力计算后的上下文向量。
        - attn（可选）: 注意力权重矩阵。
        """
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        # 转置维度以适配计算
        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        # 计算采样和稀疏选择参数
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # 采样键的数量
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # 稀疏查询的数量

        U_part = min(U_part, L_K)
        u = min(u, L_Q)

        # 稀疏注意力得分计算
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # 缩放注意力得分
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        # 初始化上下文
        context = self._get_initial_context(values, L_Q)

        # 更新上下文
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        """
        多头注意力层的实现。

        参数：
        - attention: 内部的注意力机制（如 FullAttention 或 ProbAttention）。
        - d_model: 输入特征的维度。
        - n_heads: 多头注意力的头数。
        - d_keys: 每个注意力头中键（Key）的维度，默认为 d_model // n_heads。
        - d_values: 每个注意力头中值（Value）的维度，默认为 d_model // n_heads。
        """
        super(AttentionLayer, self).__init__()

        # 如果未指定 d_keys 和 d_values，默认按头数划分维度
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention  # 使用的注意力机制（如 ProbAttention）
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)  # 投影到查询向量的维度
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)  # 投影到键向量的维度
        self.value_projection = nn.Linear(d_model, d_values * n_heads)  # 投影到值向量的维度
        self.out_projection = nn.Linear(d_values * n_heads, d_model)  # 输出投影
        self.n_heads = n_heads  # 多头注意力的头数

    def forward(self, queries, keys, values, attn_mask):
        """
        前向传播函数。

        参数：
        - queries: 查询向量，形状为 (B, L, d_model)。
        - keys: 键向量，形状为 (B, S, d_model)。
        - values: 值向量，形状为 (B, S, d_model)。
        - attn_mask: 注意力掩码，用于屏蔽某些位置。

        返回：
        - out: 注意力计算的输出，形状为 (B, L, d_model)。
        - attn: 注意力权重（如果内部注意力机制支持输出）。
        """
        # 获取输入的形状
        B, L, _ = queries.shape  # B: 批量大小, L: 查询序列长度
        _, S, _ = keys.shape  # S: 键序列长度
        H = self.n_heads  # 多头注意力的头数

        # 投影查询、键和值向量，并将它们拆分为多头
        queries = self.query_projection(queries).view(B, L, H, -1)  # (B, L, H, d_keys)
        keys = self.key_projection(keys).view(B, S, H, -1)  # (B, S, H, d_keys)
        values = self.value_projection(values).view(B, S, H, -1)  # (B, S, H, d_values)

        # 使用内部注意力机制计算输出和注意力权重
        out, attn = self.inner_attention(
            queries,  # 查询向量
            keys,  # 键向量
            values,  # 值向量
            attn_mask  # 注意力掩码
        )

        # 将多头输出重新组合到单个张量中
        out = out.view(B, L, -1)  # (B, L, d_model)

        # 投影回原始的 d_model 维度
        return self.out_projection(out), attn
