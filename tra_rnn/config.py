"""
轨迹RNN模型的参数控制类
"""

import torch
import os

class Config():
    """
    轨迹LSTM模型参数类
    """
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')  # 设备选择
        self.use_gpu = True if torch.cuda.is_available() else False  # 设备判断

        self.data_base_path = '..\\data\\'  # 数据集基础路径
        self.sava_base_path = '..\\save\\rnn_save\\'  # 模型结果保存基本路径

        self.log = True  # 是否记录模型训练、验证和推理日志

        self.save_trained_model = True  # 是否保存训练模型
        self.save_trained_model_type = 'BEST'

        self.pad_token = 0  # 填充符标志位
        self.loss_calculate_mode = 'mean'  # 损失计算方式，是总和（sum）还是平均（mean）
        self.classes = 2  # 标签类别

        self.epochs = 100
        self.batch_size = 32

        self.train_ratio = 0.8
        self.valid_ratio = 0.05
        self.infer_ratio = 0.15

        self.bi_lstm = True
        self.n_layers = 4
        self.dropout = 0.5
        self.hidden_size = 128

        self.lr = 0.001
        self.grad_clip = 50.0