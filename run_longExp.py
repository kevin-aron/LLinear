import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

fix_seed = 2220
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# 是否进行训练
parser.add_argument('--is_training', type=int, default=1, help='status')
# 模型前缀
parser.add_argument('--model_id', type=str, default='test', help='model id')
# 选择模型(可选模型有Autoformer, Informer, Transformer，DLinear，NLinear)
parser.add_argument('--model', type=str, default='DLinear',
                    help='model name, options: [Autoformer, Informer, Transformer]')

# 数据选择
parser.add_argument('--data', type=str, default='Custom', help='dataset type')
# 数据存放路径
parser.add_argument('--root_path', type=str, default='K:/LLinear/data/', help='root path of the data file')
# 数据完整名称
parser.add_argument('--data_path', type=str, default='dataflow.csv', help='data file')
# 预测类型(多变量预测、单变量预测、多元预测单变量)
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
# 如果选择单变量预测或多元预测单变量，需要指定预测的列
parser.add_argument('--target', type=str, default='flow', help='target feature in S or MS task')
# 数据重采样格式
parser.add_argument('--freq', type=str, default='m',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
# 模型存放文件夹
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# 时间窗口长度
parser.add_argument('--seq_len', type=int, default=48, help='input sequence length')
# 先验序列长度
parser.add_argument('--label_len', type=int, default=24, help='start token length')
# 要预测的序列长度
parser.add_argument('--pred_len', type=int, default=48, help='prediction sequence length')


# 针对DLinear是否为每个变量（通道）单独建立一个线性层
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

# 嵌入策略选择
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
# 编码器default参数为特征列数
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
# 解码器default参数与编码器相同
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
# 模型宽度
parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
# 多头注意力机制头数
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
# 模型中encoder层数
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
# 模型中decoder层数
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
# 全连接层神经元个数
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
# 窗口平均线的窗口大小
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
# 采样因子数
parser.add_argument('--factor', type=int, default=1, help='attn factor')
# 是否需要序列长度衰减
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
# drop_out率
parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
# 时间特征编码方式
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
# 激活函数
parser.add_argument('--activation', type=str, default='gelu', help='activation')
# 是否输出attention
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
# 是否进行预测
parser.add_argument('--do_predict', action='store_false', help='whether to predict unseen future data')

# 多线程
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
# 训练轮数
parser.add_argument('--itr', type=int, default=1, help='experiments times')
# 训练迭代次数
parser.add_argument('--train_epochs', type=int, default=1000, help='train epochs')
# batch_size大小
parser.add_argument('--batch_size', type=int, default=4, help='batch size of train input data')
# early stopping检测间隔
parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
# 学习率
parser.add_argument('--learning_rate', type=float, default=0.01, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
# loss函数
parser.add_argument('--loss', type=str, default='mse', help='loss function')
# 学习率衰减参数
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
# 是否使用自动混合精度训练
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# 是否使用GPU训练
parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
# GPU分布式训练
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
# 多GPU训练
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

# 取参数表
args = parser.parse_args()
# 获取GPU
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_rate {}'.format(args.model, args.learning_rate)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_model'.format(args.model)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
