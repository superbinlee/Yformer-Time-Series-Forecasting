import math
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        位置编码模块，基于正弦和余弦函数生成固定位置嵌入，用于序列模型中添加位置信息。

        参数：
        - d_model: 特征维度。
        - max_len: 最大支持的序列长度。
        """
        super(PositionalEmbedding, self).__init__()

        # 初始化一个用于存储位置编码的张量，大小为 (max_len, d_model)
        pe = torch.zeros(max_len, d_model).float()
        # 禁止对位置编码梯度更新（不参与训练）
        pe.require_grad = False

        # 创建一个形状为 (max_len, 1) 的位置索引张量，每个位置对应一个整数索引
        position = torch.arange(0, max_len).float().unsqueeze(1)

        # 计算 div_term，用于控制正弦和余弦的频率变化
        # div_term 是一个指数衰减序列，与 d_model 相关联
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        # 对位置编码的偶数维度赋值为正弦函数
        pe[:, 0::2] = torch.sin(position * div_term)
        # 对位置编码的奇数维度赋值为余弦函数
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加一个维度，方便后续与输入张量广播操作 (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        # 注册 `pe` 为模型的 buffer，不会被更新，但可以在设备间移动
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        前向传播函数，为输入张量添加位置编码。

        参数：
        - x: 输入张量，形状为 (B, L, D)，其中：
          - B: 批量大小。
          - L: 序列长度。
          - D: 特征维度。

        返回：
        - 位置编码张量，形状为 (1, L, D)，对应输入序列的前 L 个位置。
        """
        # 从位置编码中提取与输入序列长度匹配的部分 (1, L, D)
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        """
        TokenEmbedding 模块，用于将输入序列的每个时间步的特征映射到更高维度。
        通过一维卷积实现嵌入，相当于时间序列数据中的特征嵌入层。

        参数：
        - c_in: 输入特征维度（特征数量）。
        - d_model: 输出特征维度（嵌入维度）。
        """
        super(TokenEmbedding, self).__init__()

        # 根据 PyTorch 版本选择填充方式
        # 如果版本 >= 1.5.0，卷积填充方式为 1；否则为 2。
        padding = 1 if torch.__version__ >= '1.5.0' else 2

        # 定义一维卷积层，用于嵌入特征
        # 参数：
        # - in_channels: 输入通道数 (c_in)。
        # - out_channels: 输出通道数 (d_model)。
        # - kernel_size: 卷积核大小（3）。
        # - padding: 填充大小，确保输入和输出序列长度一致。
        # - padding_mode: 'circular' 表示循环填充，用于时间序列建模。
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode='circular'
        )

        # 初始化卷积层的权重，使用 Kaiming 正态分布初始化
        for m in self.modules():  # 遍历模块中的所有子模块
            if isinstance(m, nn.Conv1d):  # 如果是 Conv1d 层
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        """
        前向传播函数。

        参数：
        - x: 输入张量，形状为 (B, L, c_in)，其中：
          - B: 批量大小。
          - L: 序列长度。
          - c_in: 输入特征维度。

        返回：
        - x: 嵌入后的张量，形状为 (B, L, d_model)，其中 d_model 是输出嵌入维度。
        """
        # 调整输入张量的形状以适配 Conv1d
        # 从 (B, L, c_in) 转换为 (B, c_in, L)
        x = self.tokenConv(x.permute(0, 2, 1))

        # 将卷积输出的形状从 (B, d_model, L) 转换回 (B, L, d_model)
        x = x.transpose(1, 2)

        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        """
        FixedEmbedding 类，用于生成固定的嵌入层，不参与训练。
        通过正弦和余弦函数生成嵌入，类似于固定位置编码。

        参数：
        - c_in: 输入的类别数量（或序列长度，用于构建固定嵌入矩阵）。
        - d_model: 嵌入维度。
        """
        super(FixedEmbedding, self).__init__()

        # 初始化嵌入矩阵，大小为 (c_in, d_model)
        w = torch.zeros(c_in, d_model).float()
        # 禁止对嵌入矩阵梯度更新
        w.require_grad = False

        # 生成位置索引，形状为 (c_in, 1)
        position = torch.arange(0, c_in).float().unsqueeze(1)

        # 生成用于控制正弦和余弦频率变化的因子
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        # 对嵌入矩阵的偶数维度赋值为正弦函数
        w[:, 0::2] = torch.sin(position * div_term)
        # 对嵌入矩阵的奇数维度赋值为余弦函数
        w[:, 1::2] = torch.cos(position * div_term)

        # 定义一个嵌入层，将生成的固定嵌入矩阵赋值为其权重
        self.emb = nn.Embedding(c_in, d_model)
        # 使用生成的嵌入矩阵 w 初始化嵌入层权重，且不参与训练
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        """
        前向传播函数。

        参数：
        - x: 输入索引，形状为 (B, L)，其中：
          - B: 批量大小。
          - L: 序列长度。

        返回：
        - 嵌入后的张量，形状为 (B, L, d_model)。
        """
        # 返回嵌入结果，并使用 .detach() 确保梯度不会传播到嵌入权重
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        """
        TemporalEmbedding 类，用于为时间特征（如小时、星期、天、月等）生成嵌入。
        根据指定的嵌入类型（固定或可学习）创建嵌入层。

        参数：
        - d_model: 嵌入维度。
        - embed_type: 嵌入类型，'fixed' 表示固定嵌入，其他值表示可学习嵌入。
        - freq: 时间粒度，'h' 表示小时粒度，'t' 表示分钟粒度。
        """
        super(TemporalEmbedding, self).__init__()

        # 时间特征的范围大小
        minute_size = 4  # 每小时分为 4 个时间段
        hour_size = 24  # 一天 24 小时
        weekday_size = 7  # 一周 7 天
        day_size = 32  # 假设一个月最多有 31 天（额外留一个位置）
        month_size = 13  # 一年 12 个月（额外留一个位置）

        # 根据嵌入类型选择嵌入实现：固定嵌入或可学习嵌入
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding

        # 如果时间粒度为分钟（'t'），创建分钟嵌入
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)  # 嵌入层的输入大小为 minute_size，输出为 d_model

        # 创建其他时间特征的嵌入层
        self.hour_embed = Embed(hour_size, d_model)  # 小时嵌入
        self.weekday_embed = Embed(weekday_size, d_model)  # 星期嵌入
        self.day_embed = Embed(day_size, d_model)  # 天嵌入
        self.month_embed = Embed(month_size, d_model)  # 月嵌入

    def forward(self, x):
        """
        前向传播函数。

        参数：
        - x: 输入时间特征张量，形状为 (B, L, 5)，其中：
          - B: 批量大小。
          - L: 序列长度。
          - 最后一个维度表示时间特征顺序：[month, day, weekday, hour, (minute)]。

        返回：
        - 时间嵌入张量，形状为 (B, L, d_model)。
        """
        # 确保输入是整型
        x = x.long()

        # 根据时间特征的索引提取对应的嵌入
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.  # 分钟嵌入（如果存在）
        hour_x = self.hour_embed(x[:, :, 3])  # 小时嵌入
        weekday_x = self.weekday_embed(x[:, :, 2])  # 星期嵌入
        day_x = self.day_embed(x[:, :, 1])  # 天嵌入
        month_x = self.month_embed(x[:, :, 0])  # 月嵌入

        # 返回所有嵌入的加和结果
        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        """
        TimeFeatureEmbedding 类，用于对时间特征进行嵌入。
        通过一个线性层将时间特征映射到高维嵌入空间。

        参数：
        - d_model: 嵌入维度。
        - embed_type: 嵌入类型（当前未实际使用，默认值为 'timeF'）。
        - freq: 时间特征的粒度，对应输入的时间特征数量。
        """
        super(TimeFeatureEmbedding, self).__init__()

        # 定义时间特征频率映射表
        # freq_map 的值表示输入时间特征的维度数 (d_inp)
        freq_map = {
            'h': 4,  # 小时级别：包含月、日、星期、小时
            't': 5,  # 分钟级别：包含月、日、星期、小时、分钟
            's': 6,  # 秒级别：包含月、日、星期、小时、分钟、秒
            'm': 1,  # 月级别：只包含月
            'a': 1,  # 年级别（例如季度分析）：只包含一个时间特征
            'w': 2,  # 周级别：包含星期、周
            'd': 3,  # 天级别：包含月、日、星期
            'b': 3  # 工作日级别：包含月、日、星期
        }

        # 根据频率 freq 获取输入时间特征的维度数 (d_inp)
        d_inp = freq_map[freq]

        # 定义一个线性层，将时间特征维度映射到嵌入维度 d_model
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x):
        """
        前向传播函数。

        参数：
        - x: 输入时间特征张量，形状为 (B, L, d_inp)，其中：
          - B: 批量大小。
          - L: 序列长度。
          - d_inp: 时间特征维度，由 freq_map[freq] 决定。

        返回：
        - 嵌入后的张量，形状为 (B, L, d_model)。
        """
        # 使用线性层对输入时间特征进行嵌入
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        """
        DataEmbedding 类，用于对输入数据进行多种嵌入，包括值嵌入、位置嵌入和时间嵌入。
        嵌入后的数据可以更好地适配 Transformer 模型。

        参数：
        - c_in: 输入特征维度（时间序列的特征数量）。
        - d_model: 输出嵌入维度。
        - embed_type: 时间嵌入类型，'fixed' 表示固定嵌入，'timeF' 表示时间特征嵌入。
        - freq: 时间频率，用于时间嵌入的特征（如 'h' 表示小时）。
        - dropout: Dropout 概率。
        """
        super(DataEmbedding, self).__init__()

        # 值嵌入：使用 TokenEmbedding 将输入特征映射到 d_model 维度
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)

        # 位置嵌入：使用 PositionalEmbedding 为每个时间步添加位置编码
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        # 时间嵌入：根据 embed_type 和 freq 决定使用 TemporalEmbedding 或 TimeFeatureEmbedding
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != 'timeF' else
            TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )

        # Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        前向传播函数。

        参数：
        - x: 输入数据张量，形状为 (B, L, c_in)，其中：
          - B: 批量大小。
          - L: 序列长度。
          - c_in: 输入特征维度。
        - x_mark: 时间特征张量，形状为 (B, L, time_features)，其中：
          - time_features 的数量由时间嵌入类型决定。

        返回：
        - 嵌入后的张量，形状为 (B, L, d_model)。
        """
        # 计算值嵌入、位置嵌入和时间嵌入的加和
        x = (
                self.value_embedding(x) +  # 值嵌入
                self.position_embedding(x) +  # 位置嵌入
                self.temporal_embedding(x_mark)  # 时间嵌入
        )

        # 应用 Dropout
        return self.dropout(x)
