import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_ECL_hour
from exp.exp_basic import Exp_Basic
from models.model import Informer, Yformer, Yformer_skipless
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate

warnings.filterwarnings('ignore')


class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'informer': Informer,
            'yformer': Yformer,
            'yformer_skipless': Yformer_skipless

        }
        if self.args.model == 'informer' or self.args.model == "yformer" or self.args.model == "yformer_skipless":
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in,
                self.args.c_out,
                self.args.seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.factor,
                self.args.d_model,
                self.args.n_heads,
                self.args.e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.dropout,
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.device
            ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        """
        根据任务标志 (flag) 和用户参数初始化数据集和数据加载器。

        参数：
        - flag: 字符串，表示任务类型，可选值为 'train', 'test', 或 'pred'。

        返回：
        - data_set: 数据集对象。
        - data_loader: 数据加载器。
        """
        # 获取用户参数
        args = self.args

        # 定义数据集映射表，指定不同数据类型对应的处理类
        data_dict = {
            'ETTh1': Dataset_ETT_hour,  # ETT 小时级别数据集 1
            'ETTh2': Dataset_ETT_hour,  # ETT 小时级别数据集 2
            'ETTm1': Dataset_ETT_minute,  # ETT 分钟级别数据集 1
            'ETTm2': Dataset_ETT_minute,  # ETT 分钟级别数据集 2
            'ECL': Dataset_ECL_hour,  # ECL 数据集（小时级别）
            'custom': Dataset_Custom,  # 自定义数据集
        }

        # 根据用户参数选择对应的数据集类
        Data = data_dict[self.args.data]

        # 判断时间编码方式：如果嵌入方式为 'timeF'，则 timeenc=1；否则为 0
        timeenc = 0 if args.embed != 'timeF' else 1

        # 根据任务类型 flag 设置数据加载器的参数
        if flag == 'test':  # 测试集配置
            shuffle_flag = False  # 测试集不需要打乱顺序
            drop_last = True  # 丢弃最后一个不完整批次
            batch_size = args.batch_size  # 使用用户定义的批量大小
            freq = args.freq  # 数据频率（如 'h' 表示小时，'t' 表示分钟）
        elif flag == 'pred':  # 预测任务配置
            shuffle_flag = False  # 预测任务不需要打乱顺序
            drop_last = False  # 不丢弃最后一个批次
            batch_size = 1  # 每次预测一个样本
            freq = args.detail_freq  # 使用更细粒度的频率
            Data = Dataset_Pred  # 使用专门的预测数据集类
        else:  # 训练集配置
            shuffle_flag = True  # 训练集需要打乱顺序
            drop_last = True  # 丢弃最后一个不完整批次
            batch_size = args.batch_size  # 使用用户定义的批量大小
            freq = args.freq  # 数据频率

        # 初始化数据集
        data_set = Data(
            root_path=args.root_path,  # 数据集的根路径
            data_path=args.data_path,  # 数据文件路径
            flag=flag,  # 当前任务标志（'train', 'test', 'pred'）
            size=[args.seq_len, args.label_len, args.pred_len],  # 输入、标签和预测序列长度
            features=args.features,  # 输入特征类型（如 'M' 表示多变量，'S' 表示单变量）
            target=args.target,  # 目标变量名
            use_decoder_tokens=args.use_decoder_tokens,  # 是否使用解码器输入
            timeenc=timeenc,  # 时间编码类型
            freq=freq  # 时间频率
        )

        # 打印任务标志和数据集大小
        print(flag, len(data_set))

        # 初始化数据加载器
        data_loader = DataLoader(
            data_set,  # 数据集
            batch_size=batch_size,  # 批量大小
            shuffle=shuffle_flag,  # 是否打乱顺序
            num_workers=args.num_workers,  # 数据加载的线程数
            drop_last=drop_last  # 是否丢弃最后一个不完整批次
        )

        # 返回数据集对象和数据加载器
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):  # 验证函数，用于评估模型在验证集上的表现
        self.model.eval()  # 设置模型为评估模式，禁用 dropout 和 batch normalization
        total_loss = []  # 存储每个 batch 的损失
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):  # 遍历验证集数据加载器
            batch_x = batch_x.float().to(self.device)  # 将输入数据 batch_x 转为 float 类型并加载到指定设备
            batch_y = batch_y.float()  # 将目标数据 batch_y 转为 float 类型
            batch_x_mark = batch_x_mark.float().to(self.device)  # 时间戳标记 batch_x_mark 转为 float 并加载到设备
            batch_y_mark = batch_y_mark.float().to(self.device)  # 时间戳标记 batch_y_mark 转为 float 并加载到设备

            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()  # 初始化解码器输入为零张量，形状与目标预测部分一致

            if self.args.use_decoder_tokens:  # 如果解码器需要使用之前时间步的标记
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)  # 拼接目标前 label_len 部分和零张量
            else:
                dec_inp = dec_inp.float().to(self.device)  # 否则直接将零张量加载到设备

            if self.args.use_amp:  # 如果启用了自动混合精度
                with torch.cuda.amp.autocast():  # 自动混合精度上下文
                    if self.args.output_attention:  # 如果需要输出注意力
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]  # 获取模型输出
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[:, -self.args.pred_len:, :]  # 获取最后预测部分输出
            else:  # 如果未启用混合精度
                if self.args.output_attention:  # 如果需要输出注意力
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]  # 获取模型输出
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[:, -self.args.pred_len:, :]  # 获取最后预测部分输出

            f_dim = -1 if self.args.features == 'MS' else 0  # 确定目标维度，'MS' 时选择最后一维，否则选择第 0 维
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)  # 选取目标序列预测部分并加载到设备

            pred = outputs.detach().cpu()  # 将预测结果从计算图中分离并转移到 CPU
            true = batch_y.detach().cpu()  # 将真实值从计算图中分离并转移到 CPU

            loss = criterion(pred, true)  # 计算当前 batch 的损失
            total_loss.append(loss)  # 将损失添加到 total_loss 列表中

        total_loss = np.average(total_loss)  # 计算平均损失
        self.model.train()  # 恢复模型为训练模式
        return total_loss  # 返回验证集平均损失

    def train(self, setting):  # 训练函数，执行训练、验证和测试的完整过程
        train_data, train_loader = self._get_data(flag='train')  # 获取训练数据和数据加载器
        vali_data, vali_loader = self._get_data(flag='val')  # 获取验证数据和数据加载器
        test_data, test_loader = self._get_data(flag='test')  # 获取测试数据和数据加载器
        path = os.path.join(self.args.checkpoints, setting)  # 定义模型保存路径
        if not os.path.exists(path): os.makedirs(path)  # 如果路径不存在，创建路径
        time_now = time.time()  # 记录当前时间
        train_steps = len(train_loader)  # 获取训练数据的总步数
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)  # 初始化早停机制
        model_optim = self._select_optimizer()  # 选择优化器
        criterion = self._select_criterion()  # 选择损失函数
        if self.args.use_amp: scaler = torch.cuda.amp.GradScaler()  # 如果使用自动混合精度，初始化 GradScaler

        for epoch in range(self.args.train_epochs):  # 迭代每个训练轮次
            iter_count, train_loss, auto_train_loss, combined_train_loss = 0, [], [], []  # 初始化迭代计数器和损失记录
            self.model.train()  # 设置模型为训练模式
            epoch_time = time.time()  # 记录当前轮次的起始时间
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):  # 遍历训练数据加载器
                iter_count += 1  # 更新迭代计数器
                model_optim.zero_grad()  # 清除优化器的梯度
                batch_x, batch_y = batch_x.float().to(self.device), batch_y.float()  # 加载输入数据和目标数据到设备
                batch_x_mark, batch_y_mark = batch_x_mark.float().to(self.device), batch_y_mark.float().to(self.device)  # 加载时间标记到设备
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()  # 初始化解码器输入为零张量
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device) if self.args.use_decoder_tokens else dec_inp.float().to(self.device)  # 如果启用解码器标记，拼接标记
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0] if self.args.output_attention else self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)  # 根据设置选择是否输出注意力
                f_dim = -1 if self.args.features == 'MS' else 0  # 确定目标特征维度
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)  # 获取目标序列的预测部分
                auto_loss = criterion(outputs[:, :-self.args.pred_len, :], batch_x)  # 计算自回归损失
                auto_train_loss.append(auto_loss.item())  # 记录自回归损失
                loss = criterion(outputs[:, -self.args.pred_len:, :], batch_y)  # 计算预测损失
                train_loss.append(loss.item())  # 记录预测损失
                combined_loss = self.args.alpha * auto_loss + (1 - self.args.alpha) * loss  # 计算组合损失
                combined_train_loss.append(combined_loss.item())  # 记录组合损失
                if (i + 1) % 100 == 0:  # 每 100 步打印一次日志
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f} | auto loss: {3:.7f} | comb loss: {4:.7f}".format(i + 1, epoch + 1, loss.item(), auto_loss.item(), combined_loss.item()))
                    speed = (time.time() - time_now) / iter_count  # 计算训练速度
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)  # 估算剩余时间
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))  # 打印速度和剩余时间
                    iter_count, time_now = 0, time.time()  # 重置计数器和时间记录
                combined_loss.backward()  # 反向传播计算梯度
                model_optim.step()  # 更新模型参数
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))  # 打印当前轮次的耗时
            train_loss, auto_loss, combined_loss = np.average(train_loss), np.average(auto_train_loss), np.average(combined_train_loss)  # 计算平均损失
            vali_loss, test_loss = self.vali(vali_data, vali_loader, criterion), self.vali(test_data, test_loader, criterion)  # 验证和测试损失
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} | Auto Loss : {3:.7f} | Comb Loss : {4:.7f}, Vali Loss: {5:.7f} Test Loss: {6:.7f}".format(epoch + 1, train_steps, train_loss, auto_loss, combined_loss, vali_loss, test_loss))  # 打印损失信息
            early_stopping(vali_loss, self.model, path)  # 检查早停条件
            if early_stopping.early_stop: print("Early stopping"); break  # 如果满足早停条件，停止训练
            adjust_learning_rate(model_optim, epoch + 1, self.args)  # 调整学习率
        best_model_path = path + '/' + 'checkpoint.pth'  # 加载最优模型的保存路径
        self.model.load_state_dict(torch.load(best_model_path))  # 加载最优模型参数
        return self.model  # 返回训练好的模型

    def test(self, setting):  # 测试函数，用于评估模型在测试集上的性能
        test_data, test_loader = self._get_data(flag='test')  # 获取测试数据和数据加载器
        self.model.eval()  # 将模型设置为评估模式，禁用 dropout 和 batch normalization
        preds, trues = [], []  # 初始化存储预测值和真实值的列表

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):  # 遍历测试数据加载器
            batch_x, batch_y = batch_x.float().to(self.device), batch_y.float()  # 将输入数据和目标数据加载到设备
            batch_x_mark, batch_y_mark = batch_x_mark.float().to(self.device), batch_y_mark.float().to(self.device)  # 加载时间标记到设备
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()  # 初始化解码器输入为零张量
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device) if self.args.use_decoder_tokens else dec_inp.float().to(self.device)  # 根据设置拼接解码器输入

            if self.args.use_amp:  # 如果启用自动混合精度
                with torch.cuda.amp.autocast():  # 在混合精度上下文中执行模型
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0] if self.args.output_attention else self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[:, -self.args.pred_len:, :]  # 获取输出
            else:  # 未启用混合精度
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0] if self.args.output_attention else self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[:, -self.args.pred_len:, :]  # 获取输出

            f_dim = -1 if self.args.features == 'MS' else 0  # 确定目标特征维度，'MS' 表示最后一维，其他情况为第 0 维
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)  # 获取目标序列的预测部分
            pred, true = outputs.detach().cpu().numpy(), batch_y.detach().cpu().numpy()  # 将预测结果和真实值从计算图中分离并转为 NumPy 数组
            preds.append(pred)  # 保存预测结果
            trues.append(true)  # 保存真实值

        preds, trues = np.array(preds), np.array(trues)  # 将预测值和真实值转换为 NumPy 数组
        print('test shape:', preds.shape, trues.shape)  # 打印测试数据形状
        preds, trues = preds.reshape(-1, preds.shape[-2], preds.shape[-1]), trues.reshape(-1, trues.shape[-2], trues.shape[-1])  # 调整数组形状
        print('test shape:', preds.shape, trues.shape)  # 打印调整后的形状
        folder_path = './results/' + setting + '/'  # 定义结果保存路径
        if not os.path.exists(folder_path): os.makedirs(folder_path)  # 如果路径不存在，则创建
        mae, mse, rmse, mape, mspe = metric(preds, trues)  # 计算评估指标
        print('mse:{}, mae:{}'.format(mse, mae))  # 打印 MSE 和 MAE
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))  # 保存评估指标
        np.save(folder_path + 'pred.npy', preds)  # 保存预测结果
        np.save(folder_path + 'true.npy', trues)  # 保存真实值
        return  # 返回


def predict(self, setting, load=False):  # 预测函数，用于生成未来数据的预测值
    pred_data, pred_loader = self._get_data(flag='pred')  # 获取预测数据和数据加载器
    if load:  # 如果需要加载模型权重
        path = os.path.join(self.args.checkpoints, setting)  # 获取模型检查点路径
        best_model_path = path + '/' + 'checkpoint.pth'  # 拼接检查点文件路径
        self.model.load_state_dict(torch.load(best_model_path))  # 加载模型权重
    self.model.eval()  # 将模型设置为评估模式
    preds = []  # 初始化预测结果列表

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):  # 遍历预测数据加载器
        batch_x, batch_y = batch_x.float().to(self.device), batch_y.float()  # 将输入数据和目标数据加载到设备
        batch_x_mark, batch_y_mark = batch_x_mark.float().to(self.device), batch_y_mark.float().to(self.device)  # 加载时间标记到设备
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()  # 初始化解码器输入为零张量
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device) if self.args.use_decoder_tokens else dec_inp.float().to(self.device)  # 根据设置拼接解码器输入
        if self.args.use_amp:  # 如果启用自动混合精度
            with torch.cuda.amp.autocast():  # 在混合精度上下文中执行模型
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0] if self.args.output_attention else self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)  # 获取输出
        else:  # 未启用混合精度
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0] if self.args.output_attention else self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)  # 获取输出
        f_dim = -1 if self.args.features == 'MS' else 0  # 确定目标特征维度，'MS' 表示最后一维，其他情况为第 0 维
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)  # 获取目标序列的预测部分
        pred = outputs.detach().cpu().numpy()  # 将预测结果从计算图中分离并转为 NumPy 数组
        preds.append(pred)  # 保存预测结果

    preds = np.array(preds)  # 将预测列表转为 NumPy 数组
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])  # 调整预测结果的形状
    folder_path = './results/' + setting + '/'  # 定义结果保存路径
    if not os.path.exists(folder_path): os.makedirs(folder_path)  # 如果路径不存在，则创建
    np.save(folder_path + 'real_prediction.npy', preds)  # 保存预测结果为 NumPy 文件
    return  # 返回
