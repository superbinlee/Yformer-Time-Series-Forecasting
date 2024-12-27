import torch
import torch.nn as nn

from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.decoder import Decoder, DecoderLayer, YformerDecoderLayer, YformerDecoder, DeConvLayer, YformerDecoder_skipless, YformerDecoderLayer_skipless
from models.embed import DataEmbedding
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack, YformerEncoder

debug = False


class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True,
                 device=torch.device('cuda:0')):
        """
        Informer 模型，用于高效长序列时间序列预测。

        参数：
        - enc_in: 编码器输入特征维度。
        - dec_in: 解码器输入特征维度。
        - c_out: 输出特征维度。
        - seq_len: 输入序列长度。
        - label_len: 解码器上下文序列长度。
        - out_len: 输出序列长度（预测长度）。
        - factor: 稀疏注意力采样因子。
        - d_model: 模型隐藏层维度。
        - n_heads: 多头注意力的头数。
        - e_layers: 编码器层数。
        - d_layers: 解码器层数。
        - d_ff: 前馈网络的维度。
        - dropout: Dropout 比例。
        - attn: 注意力类型（'prob' 或 'full'）。
        - embed: 嵌入方式（'fixed' 或 'learned'）。
        - freq: 时间特征频率（如 'h' 表示小时）。
        - activation: 激活函数（如 'gelu'）。
        - output_attention: 是否输出注意力权重。
        - distil: 是否使用蒸馏机制（减少序列长度）。
        - device: 模型运行设备。
        """
        super(Informer, self).__init__()
        self.pred_len = out_len  # 预测序列长度
        self.attn = attn  # 注意力机制类型
        self.output_attention = output_attention  # 是否输出注意力权重

        # 编码器和解码器的嵌入层
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        # 根据注意力类型选择注意力机制
        Attn = ProbAttention if attn == 'prob' else FullAttention

        # 编码器部分
        self.encoder = Encoder(
            [
                # 编码器层
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            # 蒸馏卷积层（用于减少序列长度，提高效率）
            [
                ConvLayer(d_model) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)  # 层归一化
        )

        # 解码器部分
        self.decoder = Decoder(
            [
                # 解码器层
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)  # 层归一化
        )

        # 最后的全连接投影层，将隐藏层输出映射到目标维度
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        前向传播函数。

        参数：
        - x_enc: 编码器输入序列。
        - x_mark_enc: 编码器的时间戳信息。
        - x_dec: 解码器输入序列。
        - x_mark_dec: 解码器的时间戳信息。
        - enc_self_mask: 编码器的自注意力掩码。
        - dec_self_mask: 解码器的自注意力掩码。
        - dec_enc_mask: 解码器与编码器交互的注意力掩码。

        返回：
        - dec_out: 解码器预测结果。
        - attns（可选）: 注意力权重（如果 output_attention=True）。
        """
        # 编码器嵌入和编码
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns, x_list = self.encoder(enc_out, attn_mask=enc_self_mask)

        # 解码器嵌入和解码
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        # 投影到目标维度
        dec_out = self.projection(dec_out)

        if self.output_attention:
            # 返回预测结果和注意力权重
            return dec_out[:, -self.pred_len:, :], attns
        else:
            # 只返回预测结果
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


class Yformer_skipless(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True,
                 device=torch.device('cuda:0')):
        """
        Yformer_skipless 模型的定义，适用于时间序列预测。

        参数：
        - enc_in: 编码器输入特征维度。
        - dec_in: 解码器输入特征维度。
        - c_out: 输出特征维度。
        - seq_len: 输入序列长度。
        - label_len: 解码器上下文序列长度。
        - out_len: 输出序列长度（预测长度）。
        - factor: 稀疏注意力采样因子。
        - d_model: 模型隐藏层维度。
        - n_heads: 多头注意力头数。
        - e_layers: 编码器层数。
        - d_layers: 解码器层数。
        - d_ff: 前馈网络的维度。
        - dropout: Dropout 比例。
        - attn: 注意力类型（'prob' 或 'full'）。
        - embed: 嵌入方式（'fixed' 或 'learned'）。
        - freq: 时间特征频率（如 'h' 表示小时）。
        - activation: 激活函数（如 'gelu'）。
        - output_attention: 是否输出注意力权重。
        - distil: 是否使用蒸馏机制（减少序列长度）。
        - device: 模型运行设备。
        """
        super(Yformer_skipless, self).__init__()
        self.pred_len = out_len  # 预测序列长度
        self.seq_len = seq_len  # 输入序列长度
        self.attn = attn  # 注意力类型
        self.output_attention = output_attention  # 是否输出注意力权重

        # 编码器和未来编码器的嵌入层
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.fut_enc_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        # 根据注意力类型选择注意力机制
        Attn = ProbAttention if attn == 'prob' else FullAttention

        # 编码器部分
        self.encoder = YformerEncoder(
            [
                # 创建多个编码器层（EncoderLayer）
                EncoderLayer(
                    # 注意力层 (AttentionLayer)
                    AttentionLayer(
                        # 注意力机制，使用 ProbAttention 或 FullAttention
                        Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                        d_model,  # 特征维度
                        n_heads  # 多头注意力头数
                    ),
                    d_model,  # 输入特征维度
                    d_ff,  # 前馈网络隐藏层维度
                    dropout=dropout,  # Dropout 比例
                    activation=activation  # 激活函数 (如 gelu 或 relu)
                ) for l in range(e_layers)  # 创建 e_layers 层
            ],
            # 蒸馏卷积层 (可选，用于减少序列长度)
            [
                ConvLayer(d_model) for l in range(e_layers)  # 每层一个卷积层
            ] if distil else None,  # 如果启用了蒸馏机制，则添加卷积层
            # 层归一化 (LayerNorm)
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # 未来编码器部分
        self.future_encoder = YformerEncoder(
            [
                EncoderLayer(
                    AttentionLayer(FullAttention(True, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(d_model) for l in range(e_layers)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # 解码器部分
        self.udecoder = YformerDecoder_skipless(
            [
                YformerDecoderLayer_skipless(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(d_layers)
            ],
            [
                DeConvLayer(d_model) for l in range(d_layers)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # 输出投影层，将隐藏层输出映射到目标维度
        self.seq_len_projection = nn.Linear(d_model, c_out, bias=True)  # 用于输入序列的输出投影
        self.pred_len_projection = nn.Linear(d_model, c_out, bias=True)  # 用于预测序列的输出投影

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        前向传播函数。

        参数：
        - x_enc: 编码器输入序列。
        - x_mark_enc: 编码器的时间戳信息。
        - x_dec: 解码器输入序列。
        - x_mark_dec: 解码器的时间戳信息。
        - enc_self_mask: 编码器的自注意力掩码。
        - dec_self_mask: 解码器的自注意力掩码。
        - dec_enc_mask: 解码器与编码器交互的注意力掩码。

        返回：
        - dec_out: 解码器的预测结果。
        - attns（可选）: 注意力权重（如果 output_attention=True）。
        """
        # 编码器嵌入和编码
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns, x_list = self.encoder(enc_out, attn_mask=enc_self_mask)
        x_list.reverse()  # 反转输出以供解码器使用

        # 未来编码器嵌入和编码
        fut_enc_out = self.fut_enc_embedding(x_dec, x_mark_dec)
        fut_enc_out, attns, fut_x_list = self.future_encoder(fut_enc_out, attn_mask=enc_self_mask)
        fut_x_list.reverse()

        # 解码器部分
        dec_out, attns = self.udecoder(x_list, fut_x_list, attn_mask=dec_self_mask)

        # 投影输出结果
        seq_len_dec_out = self.pred_len_projection(dec_out)[:, -(self.seq_len):, :]
        pre_len_dec_out = self.seq_len_projection(dec_out)[:, -(self.pred_len):, :]

        # 合并输入和预测部分
        dec_out = torch.cat((seq_len_dec_out, pre_len_dec_out), dim=1)

        if self.output_attention:
            # 返回预测结果和注意力权重
            return dec_out, attns
        else:
            # 只返回预测结果
            return dec_out  # [B, L, D]


class Yformer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True,
                 device=torch.device('cuda:0')):
        """
        Yformer 架构的初始化函数。

        参数：
        - enc_in: 编码器输入特征的维度。
        - dec_in: 解码器输入特征的维度。
        - c_out: 输出特征的维度。
        - seq_len: 输入序列的长度。
        - label_len: 解码器的上下文长度。
        - out_len: 输出序列的长度。
        - factor: 注意力机制中的采样因子。
        - d_model: 模型隐藏层的维度。
        - n_heads: 多头注意力的头数。
        - e_layers: 编码器的层数。
        - d_layers: 解码器的层数。
        - d_ff: 前馈网络的维度。
        - dropout: Dropout 概率。
        - attn: 注意力类型（'prob' 或 'full'）。
        - embed: 嵌入方式（'fixed' 或 'learned'）。
        - freq: 时间特征的频率（如 'h' 表示小时）。
        - activation: 激活函数（如 'gelu'）。
        - output_attention: 是否输出注意力权重。
        - distil: 是否使用蒸馏层（减少序列长度）。
        - device: 模型运行的设备。
        """
        super(Yformer, self).__init__()
        self.pred_len = out_len  # 预测序列长度
        self.seq_len = seq_len  # 输入序列长度
        self.attn = attn  # 注意力机制类型
        self.output_attention = output_attention  # 是否输出注意力权重

        # 编码器和未来编码器的嵌入层
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.fut_enc_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        # 根据注意力类型选择注意力机制
        Attn = ProbAttention if attn == 'prob' else FullAttention

        # 编码器部分，使用 ProbSparse 注意力
        self.encoder = YformerEncoder(
            attn_layers=[
                # 创建 e_layers 个编码器层，每个层包含一个注意力模块
                EncoderLayer(
                    # 注意力层
                    AttentionLayer(
                        # 注意力机制：可以是 ProbAttention 或 FullAttention
                        Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                        d_model,  # 输入特征维度
                        n_heads  # 多头注意力头数
                    ),
                    d_model,  # 输入特征维度
                    d_ff,  # 前馈网络隐藏层维度
                    dropout=dropout,  # Dropout 比例
                    activation=activation  # 激活函数 (如 "gelu" 或 "relu")
                ) for l in range(e_layers)  # 循环创建 e_layers 层
            ],
            conv_layers=[
                # 创建 e_layers 个卷积层 (用于蒸馏机制)
                ConvLayer(d_model) for l in range(e_layers)
            ] if distil else None,  # 如果启用蒸馏机制 (distil=True)，则包含卷积层
            norm_layer=torch.nn.LayerNorm(d_model)  # 归一化层
        )

        # 未来编码器部分，使用全局注意力（masked attention）
        self.future_encoder = YformerEncoder(
            [
                # 创建多个编码器层，每个层包含一个注意力模块
                EncoderLayer(
                    # 注意力层
                    AttentionLayer(
                        # 使用 FullAttention 注意力机制
                        FullAttention(True, factor, attention_dropout=dropout, output_attention=output_attention),
                        d_model,  # 输入特征维度
                        n_heads  # 多头注意力头数
                    ),
                    d_model,  # 输入特征维度
                    d_ff,  # 前馈网络隐藏层维度
                    dropout=dropout,  # Dropout 比例
                    activation=activation  # 激活函数（如 GELU 或 ReLU）
                ) for l in range(e_layers)  # 创建 e_layers 个层
            ],
            # 可选的卷积层，用于序列降维或特征提取
            [
                ConvLayer(d_model) for l in range(e_layers)  # 每层一个卷积层
            ] if distil else None,  # 如果启用蒸馏机制，则包含卷积层
            norm_layer=torch.nn.LayerNorm(d_model)  # 输出归一化层
        )

        # 解码器部分
        self.udecoder = YformerDecoder(
            attn_layers=[
                # 创建多个解码器层，每个层包含一个解码器模块
                YformerDecoderLayer(
                    # 注意力层
                    AttentionLayer(
                        # 使用 ProbAttention 或 FullAttention 注意力机制
                        Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                        d_model,  # 输入特征维度
                        n_heads  # 多头注意力头数
                    ),
                    d_model,  # 输入特征维度
                    d_ff,  # 前馈网络隐藏层维度
                    dropout=dropout,  # Dropout 比例
                    activation=activation  # 激活函数（如 GELU 或 ReLU）
                ) for l in range(d_layers)  # 创建 d_layers 个解码器层
            ],
            # 可选的反卷积层，用于序列重构或扩展
            conv_layers=[
                DeConvLayer(d_model) for l in range(d_layers)
            ] if distil else None,  # 如果启用蒸馏机制，则包含反卷积层
            norm_layer=torch.nn.LayerNorm(d_model)  # 输出归一化层
        )

        # 输出的全连接层，用于映射到目标维度
        self.seq_len_projection = nn.Linear(d_model, c_out, bias=True)
        self.pred_len_projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        前向传播函数。

        参数：
        - x_enc: 编码器的输入序列。
        - x_mark_enc: 编码器的时间戳信息。
        - x_dec: 解码器的输入序列。
        - x_mark_dec: 解码器的时间戳信息。
        - enc_self_mask: 编码器的自注意力掩码。
        - dec_self_mask: 解码器的自注意力掩码。
        - dec_enc_mask: 解码器-编码器交互的注意力掩码。

        返回：
        - dec_out: 解码器的预测结果。
        - attns（可选）: 注意力权重（如果 output_attention=True）。
        """
        # 编码器嵌入和编码
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # 将输入序列 `x_enc` 和时间戳信息 `x_mark_enc` 通过嵌入层进行高维特征转换，生成嵌入后的特征。
        enc_out, attns, x_list = self.encoder(enc_out, attn_mask=enc_self_mask)
        # 将嵌入特征输入编码器 `self.encoder` 进行多层处理，得到最终编码输出 `enc_out`、注意力权重列表 `attns` 以及每层编码器的中间输出列表 `x_list`。
        x_list.reverse()
        # 反转中间输出 `x_list` 的顺序，为解码器提供多尺度的特征表示（从高层特征到低层特征）。
        # 未来编码器嵌入和编码
        fut_enc_out = self.fut_enc_embedding(x_dec, x_mark_dec)
        # 将目标序列 `x_dec` 和时间戳信息 `x_mark_dec` 通过嵌入层 `self.fut_enc_embedding` 进行高维特征转换。
        fut_enc_out, attns, fut_x_list = self.future_encoder(fut_enc_out, attn_mask=enc_self_mask)
        # 将嵌入特征输入未来编码器 `self.future_encoder`，生成未来序列的特征表示 `fut_enc_out`、注意力权重列表 `attns` 以及每层未来编码器的中间输出列表 `fut_x_list`。
        fut_x_list.reverse()
        # 反转未来编码器中间输出 `fut_x_list` 的顺序，以便解码器逐步利用不同尺度的特征。
        # 解码器部分
        dec_out, attns = self.udecoder(x_list, fut_x_list, attn_mask=dec_self_mask)
        # 将编码器的中间输出 `x_list` 和未来编码器的中间输出 `fut_x_list` 传入解码器 `self.udecoder`，结合生成解码输出 `dec_out` 和解码器的注意力权重列表 `attns`。
        # 将解码器输出映射到预测序列长度和输入序列长度
        seq_len_dec_out = self.pred_len_projection(dec_out)[:, -(self.seq_len):, :]
        # 将解码器输出通过 `pred_len_projection` 映射到输入序列部分的特征，并截取最后 `seq_len` 长度。
        pre_len_dec_out = self.seq_len_projection(dec_out)[:, -(self.pred_len):, :]
        # 将解码器输出通过 `seq_len_projection` 映射到预测序列部分的特征，并截取最后 `pred_len` 长度。
                # Unet: 合并输入和预测部分，这里体现了Unet的思想，跳跃连接：通过 x_list 和 fut_x_list 的传递，解码器结合了编码器和未来编码器的多尺度特征。
        dec_out = torch.cat((seq_len_dec_out, pre_len_dec_out), dim=1)
        # 将输入序列特征部分和预测序列特征部分在序列维度上拼接，形成完整的解码器输出。
        if self.output_attention:
            return dec_out, attns
            # 如果启用了输出注意力，则返回解码器的最终输出 `dec_out` 和注意力权重 `attns`。
        else:
            return dec_out  # [B, L, D]
        # 如果未启用输出注意力，则只返回解码器的最终输出 `dec_out`。


class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True,
                 device=torch.device('cuda:0')):
        """
        InformerStack 架构的初始化函数。

        参数：
        - enc_in: 编码器输入特征的维度。
        - dec_in: 解码器输入特征的维度。
        - c_out: 输出特征的维度。
        - seq_len: 输入序列的长度。
        - label_len: 解码器的上下文长度。
        - out_len: 输出序列的长度。
        - factor: 注意力机制中的采样因子。
        - d_model: 模型隐藏层的维度。
        - n_heads: 多头注意力的头数。
        - e_layers: 编码器的层数。
        - d_layers: 解码器的层数。
        - d_ff: 前馈网络的维度。
        - dropout: Dropout 概率。
        - attn: 注意力类型（'prob' 或 'full'）。
        - embed: 嵌入方式（'fixed' 或 'learned'）。
        - freq: 时间特征的频率（如 'h' 表示小时）。
        - activation: 激活函数（如 'gelu'）。
        - output_attention: 是否输出注意力权重。
        - distil: 是否使用蒸馏层（减少序列长度）。
        - device: 模型运行的设备。
        """
        super(InformerStack, self).__init__()
        self.pred_len = out_len  # 预测序列长度
        self.attn = attn  # 注意力机制类型
        self.output_attention = output_attention  # 是否输出注意力权重

        # 编码器和解码器的嵌入层
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        # 根据注意力类型选择注意力机制
        Attn = ProbAttention if attn == 'prob' else FullAttention

        # 编码器部分
        stacks = list(range(e_layers, 2, -1))  # 自定义堆叠的层数
        encoders = [
            Encoder(
                [
                    # 编码器层
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                       d_model, n_heads),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                # 蒸馏卷积层
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el - 1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)  # 层归一化
            ) for el in stacks]
        self.encoder = EncoderStack(encoders)  # 编码器堆叠

        # 解码器部分
        self.decoder = Decoder(
            [
                # 解码器层
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)  # 层归一化
        )

        # 最后的全连接投影层，将隐藏层输出映射到目标维度
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        前向传播函数。

        参数：
        - x_enc: 编码器的输入序列。
        - x_mark_enc: 编码器的时间戳信息。
        - x_dec: 解码器的输入序列。
        - x_mark_dec: 解码器的时间戳信息。
        - enc_self_mask: 编码器的自注意力掩码。
        - dec_self_mask: 解码器的自注意力掩码。
        - dec_enc_mask: 解码器-编码器交互的注意力掩码。

        返回：
        - dec_out: 解码器的预测结果。
        - attns（可选）: 注意力权重（如果 output_attention=True）。
        """
        # 编码器嵌入和编码
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # 解码器嵌入和解码
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        # 投影到目标维度
        dec_out = self.projection(dec_out)

        if self.output_attention:
            # 返回预测结果和注意力权重
            return dec_out[:, -self.pred_len:, :], attns
        else:
            # 只返回预测结果
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
