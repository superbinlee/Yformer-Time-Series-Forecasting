import torch
import argparse
from exp.exp_informer import Exp_Informer  # 引入实验类 Exp_Informer，用于运行实验

parser = argparse.ArgumentParser(description='[Yformer] Long Sequences Forecasting')  # 创建 ArgumentParser 对象，用于解析命令行参数
parser.add_argument('--model', type=str, required=False, default='yformer', help='实验使用的模型，选项：[informer, yformer, yformer_skipless]')
parser.add_argument('--data', type=str, required=False, default='ETTh1', help='使用的数据集名称')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='数据文件的根目录路径')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='数据文件的名称')
parser.add_argument('--features', type=str, default='M', help='预测任务的类型，选项：[M, S, MS]；M:多变量预测多变量，S:单变量预测单变量，MS:多变量预测单变量')
parser.add_argument('--target', type=str, default='OT', help='目标特征名称（用于 S 或 MS 任务）')
parser.add_argument('--freq', type=str, default='h', help='时间特征编码的频率，选项：[s:秒, t:分钟, h:小时, d:天, b:工作日, w:周, m:月]，也可以使用更详细的频率如15min或3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='模型检查点保存位置')
parser.add_argument('--seq_len', type=int, default=48, help='Informer编码器的输入序列长度')
parser.add_argument('--label_len', type=int, default=48, help='Informer解码器的起始标记序列长度')
parser.add_argument('--pred_len', type=int, default=336, help='预测序列的长度')
parser.add_argument('--enc_in', type=int, default=7, help='编码器的输入特征维度')
parser.add_argument('--dec_in', type=int, default=7, help='解码器的输入特征维度')
parser.add_argument('--c_out', type=int, default=7, help='模型的输出特征维度')
parser.add_argument('--d_model', type=int, default=512, help='模型的隐藏层维度')
parser.add_argument('--n_heads', type=int, default=8, help='多头注意力的头数')
parser.add_argument('--e_layers', type=int, default=3, help='编码器的层数')
parser.add_argument('--d_layers', type=int, default=3, help='解码器的层数')
parser.add_argument('--d_ff', type=int, default=2048, help='前馈神经网络的维度')
parser.add_argument('--factor', type=int, default=3, help='稀疏注意力机制的因子')
parser.add_argument('--distil', action='store_false', default=True, help='是否使用蒸馏（默认使用），添加该参数表示不使用蒸馏')
parser.add_argument('--dropout', type=float, default=0.05, help='Dropout 概率')
parser.add_argument('--attn', type=str, default='prob', help='编码器使用的注意力机制，选项：[prob, full]')
parser.add_argument('--embed', type=str, default='timeF', help='时间特征编码方式，选项：[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='激活函数类型')
parser.add_argument('--output_attention', action='store_true', help='是否输出编码器的注意力权重')
parser.add_argument('--do_predict', action='store_true', help='是否对未见过的未来数据进行预测')
parser.add_argument('--weight_decay', type=float, default=0.0, help='正则化的权重衰减系数')
parser.add_argument('--alpha', type=float, default=0.7, help='学习率调整因子')
parser.add_argument('--use_decoder_tokens', type=int, default=0, help='解码器是否使用前一时间步的标记，1表示使用，0表示不使用')
parser.add_argument('--num_workers', type=int, default=0, help='数据加载器的工作线程数')
parser.add_argument('--itr', type=int, default=2, help='实验重复次数')
parser.add_argument('--train_epochs', type=int, default=10, help='训练的总轮数')
parser.add_argument('--batch_size', type=int, default=2, help='训练数据的批量大小')
parser.add_argument('--patience', type=int, default=3, help='早停机制的容忍度')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='优化器的学习率')
parser.add_argument('--des', type=str, default='test', help='实验描述')
parser.add_argument('--loss', type=str, default='mse', help='损失函数类型')
parser.add_argument('--lradj', type=str, default='type1', help='学习率调整策略')
parser.add_argument('--use_amp', action='store_true', default=False, help='是否使用自动混合精度训练')
parser.add_argument('--use_gpu', type=bool, default=True, help='是否使用 GPU')
parser.add_argument('--gpu', type=int, default=0, help='使用的 GPU 编号')
parser.add_argument('--use_multi_gpu', action='store_true', default=False, help='是否使用多 GPU')
parser.add_argument('--devices', type=str, default='0,1,2,3', help='多 GPU 的设备编号')
args = parser.parse_args()  # 解析命令行参数

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTm1': {'data': 'ETTm1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTm2': {'data': 'ETTm2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ECL': {'data': './data/ECL/ECL.txt', 'T': '320', 'M': [320, 320, 320], 'S': [1, 1, 1], 'MS': [320, 320, 1]},
}

if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']  # check the target
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

args.detail_freq = args.freq
args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)

Exp = Exp_Informer

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_tk{}_wd{}_lr{}_al{}_{}_{}'.format(args.model, args.data, args.features,
                                                                                                                        args.seq_len, args.label_len, args.pred_len,
                                                                                                                        args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, args.embed, args.distil,
                                                                                                                        args.use_decoder_tokens, args.weight_decay, args.learning_rate, args.alpha, args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)

    if args.do_predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting, True)

    torch.cuda.empty_cache()
