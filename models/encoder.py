import torch
import torch.nn as nn
import torch.nn.functional as F

debug = False


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        """
        卷积层（ConvLayer），用于序列长度的降维或特征提取。

        参数：
        - c_in: 输入通道数。
        """
        super(ConvLayer, self).__init__()

        # 一维卷积层，支持循环填充（circular padding），用于特征提取
        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=3,
            padding=2,  # 循环填充，保持序列连续性
            padding_mode='circular'
        )

        # 批归一化层，防止梯度消失或爆炸
        self.norm = nn.BatchNorm1d(c_in)

        # 激活函数，使用 ELU（指数线性单元），用于引入非线性
        self.activation = nn.ELU()

        # 最大池化层，用于降维（减小序列长度）和保留特征
        self.maxPool = nn.MaxPool1d(
            kernel_size=3,  # 窗口大小
            stride=2,  # 步幅
            padding=1  # 填充，确保维度变化平滑
        )

    def forward(self, x):
        """
        前向传播函数。

        参数：
        - x: 输入张量，形状为 (B, L, C)，其中：
          - B: 批量大小。
          - L: 序列长度。
          - C: 特征维度。

        返回：
        - x: 输出张量，形状为 (B, L', C)，其中 L' 是经过池化后的序列长度。
        """
        # 转换维度以适配卷积层 (B, L, C) -> (B, C, L)
        x = self.downConv(x.permute(0, 2, 1))

        # 批归一化
        x = self.norm(x)

        # 激活函数
        x = self.activation(x)

        # 最大池化，缩减序列长度
        x = self.maxPool(x)

        # 转换维度回原始格式 (B, C, L') -> (B, L', C)
        x = x.transpose(1, 2)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        """
        编码器层（Encoder Layer），包含注意力机制和前馈网络。

        参数：
        - attention: 自注意力机制（如 ProbAttention 或 FullAttention）。
        - d_model: 输入特征的维度。
        - d_ff: 前馈网络的隐藏层维度，默认是 `4 * d_model`。
        - dropout: Dropout 比例。
        - activation: 激活函数（"relu" 或 "gelu"）。
        """
        super(EncoderLayer, self).__init__()

        # 如果未指定 d_ff，默认设置为 4 倍的 d_model
        d_ff = d_ff or 4 * d_model

        self.attention = attention  # 自注意力机制
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)  # 第一个 1D 卷积层
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)  # 第二个 1D 卷积层
        self.norm1 = nn.LayerNorm(d_model)  # 第一层归一化，用于注意力机制的残差连接
        self.norm2 = nn.LayerNorm(d_model)  # 第二层归一化，用于前馈网络的残差连接
        self.dropout = nn.Dropout(dropout)  # Dropout 层
        self.activation = F.relu if activation == "relu" else F.gelu  # 激活函数

    def forward(self, x, attn_mask=None):
        """
        前向传播函数。

        参数：
        - x: 输入张量，形状为 (B, L, D)，其中：
          - B: 批量大小。
          - L: 序列长度。
          - D: 特征维度（与 d_model 一致）。
        - attn_mask: 注意力掩码，用于屏蔽特定位置的注意力。

        返回：
        - x: 编码器层的输出，形状为 (B, L, D)。
        - attn: 注意力权重（如果内部注意力机制支持输出）。
        """
        # 自注意力机制
        new_x, attn = self.attention(
            x, x, x,  # 查询、键和值
            attn_mask=attn_mask
        )
        # 残差连接和 Dropout
        x = x + self.dropout(new_x)

        # 第一层归一化
        y = x = self.norm1(x)

        # 前馈网络
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))  # 激活后，转置适配 Conv1D 输入
        y = self.dropout(self.conv2(y).transpose(-1, 1))  # 转置回原格式

        # 第二层归一化和残差连接
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        """
        编码器（Encoder），由多个注意力层和可选的卷积层组成。

        参数：
        - attn_layers: 注意力层列表（如由 `EncoderLayer` 组成）。
        - conv_layers: 可选的卷积层列表，用于序列降维或增强特征表示。
        - norm_layer: 可选的归一化层，用于输出的标准化。
        """
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)  # 注意力层列表
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None  # 卷积层列表
        self.norm = norm_layer  # 输出归一化层

    def forward(self, x, attn_mask=None):
        """
        前向传播函数。

        参数：
        - x: 输入张量，形状为 (B, L, D)，其中：
          - B: 批量大小。
          - L: 序列长度。
          - D: 特征维度。
        - attn_mask: 注意力掩码，用于屏蔽特定位置的注意力。

        返回：
        - x: 编码器的最终输出，形状为 (B, L, D)。
        - attns: 所有注意力层的注意力权重列表。
        - x_list: 所有注意力层和卷积层的中间输出列表。
        """
        attns = []  # 存储每一层的注意力权重
        x_list = []  # 存储每一层的中间输出

        if self.conv_layers is not None:
            # 如果存在卷积层，对注意力层和卷积层交替处理
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)  # 注意力层处理
                x = conv_layer(x)  # 卷积层处理
                x_list.append(x)  # 记录中间输出
                attns.append(attn)  # 记录注意力权重
            # 处理最后一个注意力层（没有对应的卷积层）
            x, attn = self.attn_layers[-1](x)
            x_list.append(x)  # 记录最后一层输出
            attns.append(attn)  # 记录最后一层注意力权重
        else:
            # 如果没有卷积层，仅处理注意力层
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x_list.append(x)  # 记录中间输出
                attns.append(attn)  # 记录注意力权重

        # 如果定义了归一化层，对最终输出和中间输出进行归一化
        if self.norm is not None:
            x = self.norm(x)  # 对最终输出归一化
            for i in range(len(x_list)):
                x_list[i] = self.norm(x_list[i])  # 对中间输出归一化

        return x, attns, x_list


class YformerEncoder(nn.Module):
    def __init__(self, attn_layers=None, conv_layers=None, norm_layer=None):
        """
        Yformer 编码器，支持注意力层和卷积层的组合。

        参数：
        - attn_layers: 注意力层列表（如由 `EncoderLayer` 组成）。
        - conv_layers: 卷积层列表，用于特征提取或序列长度降维。
        - norm_layer: 归一化层，用于标准化输出。
        """
        super(YformerEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers) if attn_layers is not None else None  # 注意力层列表
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None  # 卷积层列表
        self.norm = norm_layer  # 输出归一化层

    def forward(self, x, attn_mask=None):
        """
        前向传播函数。

        参数：
        - x: 输入张量，形状为 (B, L, D)，其中：
          - B: 批量大小。
          - L: 序列长度。
          - D: 特征维度。
        - attn_mask: 注意力掩码，用于屏蔽特定位置的注意力。

        返回：
        - x: 编码器的最终输出，形状为 (B, L, D)。
        - attns: 每一层的注意力权重列表（如果没有注意力层，则为 None）。
        - x_list: 每一层的中间输出列表，包括初始输入。
        """
        attns = []  # 存储每层的注意力权重
        x_list = []  # 存储每层的中间输出
        x_list.append(x)  # 将输入张量加入输出列表

        if self.conv_layers is not None:
            # 如果存在卷积层
            if self.attn_layers is not None:
                # 注意力层和卷积层交替处理
                for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                    x, attn = attn_layer(x, attn_mask=attn_mask)  # 注意力层
                    x = conv_layer(x)  # 卷积层
                    x_list.append(x)  # 记录中间输出
                    attns.append(attn)  # 记录注意力权重
            else:
                # 只有卷积层的处理流程
                for conv_layer in self.conv_layers:
                    x = conv_layer(x)  # 卷积层
                    x_list.append(x)  # 记录中间输出
                    attns.append(None)  # 没有注意力权重
        else:
            # 只有注意力层的处理流程
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x_list.append(x)  # 记录中间输出
                attns.append(attn)  # 记录注意力权重

        # 如果定义了归一化层，对输出和中间结果进行归一化
        if self.norm is not None:
            x = self.norm(x)  # 对最终输出归一化
            for i in range(len(x_list)):
                x_list[i] = self.norm(x_list[i])  # 对每层的输出归一化

        return x, attns, x_list


class EncoderStack(nn.Module):
    def __init__(self, encoders):
        """
        EncoderStack 类，支持多级编码器的堆叠结构，用于多尺度建模。

        参数：
        - encoders: 编码器列表，每个编码器负责处理不同级别的输入。
        """
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)  # 存储编码器的模块列表

    def forward(self, x, attn_mask=None):
        """
        前向传播函数。

        参数：
        - x: 输入张量，形状为 (B, L, D)，其中：
          - B: 批量大小。
          - L: 输入序列长度。
          - D: 特征维度。
        - attn_mask: 注意力掩码，用于屏蔽特定位置的注意力。

        返回：
        - x_stack: 所有编码器输出的级联结果，形状为 (B, sum(L_i), D)，其中 L_i 是每层编码器的输出序列长度。
        - attns: 每个编码器的注意力权重列表。
        """
        inp_len = x.shape[1]  # 输入序列的初始长度
        x_stack = []  # 存储每个编码器的输出
        attns = []  # 存储每个编码器的注意力权重

        for encoder in self.encoders:
            if encoder is None:
                # 如果某级编码器为空，则跳过并缩减序列长度
                inp_len = inp_len // 2
                continue
            # 使用当前编码器处理输入序列的最后部分
            x, attn = encoder(x[:, -inp_len:, :], attn_mask=attn_mask)
            x_stack.append(x)  # 记录当前编码器的输出
            attns.append(attn)  # 记录当前编码器的注意力权重
            inp_len = inp_len // 2  # 缩减输入序列长度（用于多尺度处理）

        # 将所有编码器的输出在序列长度维度上拼接
        x_stack = torch.cat(x_stack, dim=-2)

        return x_stack, attns
