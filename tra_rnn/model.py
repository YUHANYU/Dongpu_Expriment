r"""
异常轨迹的LSTM模型
B: Batch size
M: Max length
D: Embedding Dimension
"""

import torch
from torch import nn
from config import Config
config = Config()


class TrackLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, bi_lstm=True, n_layers=1, dropout=0.5):
        super(TrackLSTM, self).__init__()
        """
        轨迹LSTM模型
        :param input_size: 轨迹点的维度
        :param hidden_size: LSTM隐藏层大小
        :param bi_LSTM: LSTM模型是否双向，默认双向
        :param n_layers: LSTM层数，默认1层
        :param dropout: dropout大小，默认0.5
        """

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, hidden_size)  # 创建一个轨迹点embeddings
        # TODO embedding层是不是还要指定padding_idx=config.pad_token，这需要实验验证
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            bidirectional=bi_lstm,
            dropout=dropout
        )
        # TODO 考虑堆叠式LSTM怎么构建
        self.dense = nn.Linear(in_features=hidden_size, out_features=config.classes,
                               bias=True)  # TODO 考察最后一层加不加偏置对模型预测效果的影响
    def forward(self, input_tracks, input_tracks_length, start_hidden=None):
        """
        轨迹LSTM模型前向计算
        :param input_tracks:  轨迹集合
        :param input_tracks_length: 每条轨迹的长度
        :param start_hidden: 初始LSTM隐藏层
        :return: LSTM计算输出
        """
        input_tracks = input_tracks.transpose(0, 1)  # FIXME-20201229 模型计算有问题
        tracks_embedding = self.embedding(input_tracks)  # 得到轨迹点的embedding max_len * batch_size
        tracks_packed = torch.nn.utils.rnn.pack_padded_sequence(
            tracks_embedding, input_tracks_length)  # 压缩轨迹embedding
        output, hidden = self.lstm(tracks_packed, start_hidden)  # lstm网络学习轨迹embedding
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)  # 解压输出
        if config.bi_lstm:
            output = output[:, :, :self.hidden_size] + \
                     output[:, :, self.hidden_size:]  # 双向叠加

        output_1 = output.transpose(0, 1)
        # output = output[:, -1, :]  # FIXME 这里转化有问题
        output_2 = output_1.permute(0, 2, 1)[:, :, -1]

        output_3 = self.dense(output_2)

        return output

    def __init_lstm_hidden(self):
        # TODO 初始化LSTM隐藏层
        pass
