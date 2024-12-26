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

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()

            if self.args.use_decoder_tokens:
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            else:
                dec_inp = dec_inp.float().to(self.device)

            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[:, -self.args.pred_len:, :]
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[:, -self.args.pred_len:, :]
            f_dim = -1 if self.args.features == 'MS' else 0
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()

            loss = criterion(pred, true)

            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
        #     summary(self.model,  [batch_x.shape, batch_x_mark.shape, batch_y.shape, batch_y_mark.shape]) # show the size 
        #     break

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            auto_train_loss = []
            combined_train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()

                if self.args.use_decoder_tokens:
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                else:
                    dec_inp = dec_inp.float().to(self.device)

                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                auto_loss = criterion(outputs[:, :-self.args.pred_len, :], batch_x)
                auto_train_loss.append(auto_loss.item())
                loss = criterion(outputs[:, -self.args.pred_len:, :], batch_y)
                train_loss.append(loss.item())

                combined_loss = self.args.alpha * auto_loss + (1 - self.args.alpha) * loss
                combined_train_loss.append(combined_loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f} | auto loss: {3:.7f} | comb loss: {4:.7f}".format(i + 1, epoch + 1, loss.item(), auto_loss.item(), combined_loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                combined_loss.backward()
                model_optim.step()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            auto_loss = np.average(auto_train_loss)
            combined_loss = np.average(combined_train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} | Auto Loss : {3:.7f} | Comb Loss : {4:.7f}, Vali Loss: {5:.7f} Test Loss: {6:.7f}".format(
                epoch + 1, train_steps, train_loss, auto_loss, combined_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        self.model.eval()

        preds = []
        trues = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            if self.args.use_decoder_tokens:
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            else:
                dec_inp = dec_inp.float().to(self.device)
            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[:, -self.args.pred_len:, :]
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[:, -self.args.pred_len:, :]
            f_dim = -1 if self.args.features == 'MS' else 0
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

            pred = outputs.detach().cpu().numpy()  # .squeeze()
            true = batch_y.detach().cpu().numpy()  # .squeeze()

            preds.append(pred)
            trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()

        preds = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            if self.args.use_decoder_tokens:
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            else:
                dec_inp = dec_inp.float().to(self.device)
            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if self.args.features == 'MS' else 0
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

            pred = outputs.detach().cpu().numpy()  # .squeeze()

            preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
