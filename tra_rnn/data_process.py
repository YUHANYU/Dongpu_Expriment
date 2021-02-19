"""
轨迹数据的处理
"""

import csv
import tqdm
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from config import Config
config = Config()


def merge_read_csv(abnormal_csv, normal_csv):
    """
    合并正常轨迹和异常轨迹的两个csv,再读取出来
    :param abnormal_csv: 正常轨迹csv
    :param normal_csv: 异常轨迹csv
    :return: 合并后的轨迹字典
    """
    abnormal_csv_reader = csv.reader(open(abnormal_csv, encoding='utf-8', mode='r'))
    normal_csv_reader = csv.reader(open(normal_csv, encoding='utf-8', mode='r'))
    next(abnormal_csv_reader)
    next(normal_csv_reader)
    abnormal = [line for line in abnormal_csv_reader]
    normal = [row for row in normal_csv_reader]
    merge_track = abnormal + normal
    file_name = config.base_path + 'normal_' + str(len(normal)) + '-' + \
                'abnormal_' + str(len(abnormal))+ '.csv'
    file = open(file_name, 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(file)
    for track in merge_track:
        csv_writer.writerow(track)
    file.close()

    file_result = read_track_csv(file_name)

    return file_result


def read_track_csv(dataset):
    """
    读取轨迹数据的csv文件，获取其中的网格点
    :param dataset: 轨迹数据集，有正常的也有异常的
    :return: 返回所有轨迹的字典
    """
    csv_reader = csv.reader(open(dataset, encoding='utf-8', mode='r'))  # 读取数据csv文件
    next(csv_reader)  # 跳过第一行
    tracks = {}  # 所有轨迹字典
    for idx, row in enumerate(csv_reader):  # 遍历每条轨迹
        track = {}  # 单条轨迹字典

        track['trip_id'] = int(row[0])  # 轨迹id号
        track['latitude'] = eval(row[1])  # 轨迹纬度数组
        track['lngitude'] = eval(row[2])  # 轨迹经度数组
        row_3_list = str_2_list(row[3])  # 网格线处理
        if -1 not in row_3_list:  # 排除含-1错误网格点的网格线
            track['gridline'] = eval(row[3])  # 轨迹网格点
        else:
            continue
        track['call_type'] = row[4]  # 出租车呼叫类型
        track['taxi_id'] = row[5]  # 出租车id
        track['timestamp'] = row[6]  # 轨迹时间戳
        track['label'] = row[7]  # 轨迹异常与否的标签

        tracks[str(idx)] = track  # 将每条轨迹存储到轨迹集合字典中

    return tracks


def str_2_list(str_seq):
    """
    转化字符序列为列表序列，并将每个元素转化为int整型
    :param str_seq: 输入的字符序列
    :return: 字符序列对应的列表序列
    """
    return [int(i) for i in str_seq.lstrip('[').rstrip(']').split(',')]


def strs_2_lists(str_seq):
    """
    转化字符序列为列表序列，用于转化批数据中的字符序列
    :param tgt_seq: 输入的字符序列
    :return: 字符序列对应的列表序列
    """
    ss = []
    for s in str_seq:  # 把字符序列还原list序列
        s = s.lstrip('[').rstrip(']').replace(' ', '').split(',')
        ss.append([int(i) for i in s])

    return ss


def track_process(track_dict):
    """
    将轨迹数据字典转化成pytorch可用的数据格式
    :param trackjectory_dict: 轨迹数据字典
    :return:
    """
    tracks_lat, tracks_lng, tracks_grid_dot, tracks_label = [], [], [], []
    for key, value in track_dict.items():
        tracks_lat.append(value['latitude'])
        tracks_lng.append(value['lngitude'])
        tracks_grid_dot.append(value['gridline'])
        tracks_label.append(value['label'])

    max_grid_line_length = max([len(line) for line in tracks_grid_dot])  # 最大轨迹长度
    min_grid_dot = min([min(line) for line in tracks_grid_dot])  # 最小轨迹点
    max_grid_dot = max([max(line) for line in tracks_grid_dot])  # 最大轨迹点
    # print(max_grid_line_length, min_grid_dot, max_grid_dot)

    # 每个点减去最小轨迹点，再加1！因为第一位设置为填充符
    tracks_grid_dot = [[int(dot - min_grid_dot + 1) for dot in line]
                     for line in tracks_grid_dot]  # 训练轨迹点，减去最小轨迹点，再+1。FIXME 可能有Bug

    min_grid_dot = min([min(line) for line in tracks_grid_dot])  # 最小轨迹点
    max_grid_dot = max([max(line) for line in tracks_grid_dot]) + 1  # 最大轨迹点

    # print(max_grid_line_length, min_grid_dot, max_grid_dot)

    assert len(tracks_label) == len(tracks_grid_dot), '轨迹条数和对应的标签数不一致！'

    tracks_dot_label = []
    for i in range(len(tracks_label)):
        track_dot = tracks_grid_dot[i]
        track_label = tracks_label[i]
        track_dot_label = [str(track_dot), str(track_label)]
        # track_dot_label = [str(track_dot), track_label]

        tracks_dot_label.append(track_dot_label)

    return tracks_dot_label, max_grid_dot + 1  # 加上一个填充符


def track_merge_line_label(track_dict):
    """
    轨迹处理函数，将网格线和标签合并在一起
    :param track_dict: 所有轨迹的字典
    :return: 所有轨迹-标签的列表
    """
    return [[v['gridline'], v['label']] for _, v in track_dict.items()]


def pad_seq(seq, max_len):
    """
    构造带屏蔽位置符的序列
    :param seq:
    :param max_len:
    :return:
    """
    seq += [config.pad_token for _ in range(max_len - len(seq))]

    return seq


def seq_2_tensor(seq):
    """
    将序列转化为tensor
    :param seq:
    :return:
    """
    # seq = self.str_2_list(seq)  # 还原序列
    seq_max_len = max(len(s) for s in seq)  # 该批次序列中的最大长度
    seq_len = [len(i) for i in seq]
    # 以最大长度补齐该批次的序列并转化为tensor
    seq = Variable(torch.LongTensor([pad_seq(s, seq_max_len)
                                     for s in seq])).to(config.device)

    return seq, seq_len


def batch_2_tensor(batch_data):
    """
    将批次数据转化为指定的tensor数据
    :param batch_data: 批次数据
    :return: 批次轨迹数据和标签数据及其长度
    """
    tra_dot = strs_2_lists(batch_data[0])
    tra_label = strs_2_lists(batch_data[1])

    dot_label_pairs = sorted(
        zip(tra_dot, tra_label), key=lambda p: len(p[0]), reverse=True)  # 按照轨迹长度降序排列轨迹-标签对
    tra_dot_seqs, tra_label_seqs = zip(*dot_label_pairs)  # 解压轨迹-标签对
    tra_dot_seq, tra_dot_seq_len = seq_2_tensor(tra_dot_seqs)  # 填充得到轨迹序列，轨迹对应长度
    tra_label_seq, tra_label_seq_len = seq_2_tensor(tra_label_seqs)  # 填充得到标签序列，标签长度

    return tra_dot_seq, tra_dot_seq_len, tra_label_seq, tra_label_seq_len


def split_train_valid_infer(tracks, config):
    """
    将数据集拆分为训练-验证和测试数据
    :param normal_data: 正常轨迹数据
    :param abnormal_data: 异常轨迹数据
    :param config: 参数对象
    :return: 返回训练、验证和推理的X和Y数据
    """
    train, valid_infer = train_test_split(tracks, shuffle=True,
                                           train_size=config.train_ratio,
                                           test_size=1 - config.train_ratio)
    valid, infer = train_test_split(valid_infer, shuffle=True,
                                   train_size=config.valid_ratio,
                                   test_size=config.infer_ratio)

    return train, valid, infer


def data_save(original_data, grid, persent, data_type):
    """
    将轨迹数据以.pt的文件格式保存下来
    :param original_data: 原数据数据，list格式
    :param grid: 数据集名
    :param data_type: 保存的数据类型，默认为训练
    :return: 无
    """
    data_dict = {
        'data_length': str(len(original_data)),
        'data_contend': original_data,
    }
    data_name = config.sava_base_path + grid + persent + '\\' + data_type + '.pt'
    torch.save(data_dict, data_name)


def load_data(train, valid, infer):
    """
    加载数据，
    :param train: 训练数据
    :param valid: 验证数据
    :param infer: 测试数据
    :return:
    """
    t_grid, t_grid_label = [], []
    for i in train:
        t_grid.append(i[0])
        t_grid_label.append(i[1])

    v_grid, v_grid_label = [], []
    for i in valid:
        v_grid.append(i[0])
        v_grid_label.append(i[1])

    i_grid, i_grid_label = [], []
    for i in infer:
        i_grid.append(i[0])
        i_grid_label.append(i[1])

    dot_2_idx = {config.pad_token:0}
    grids = [t_grid, v_grid, i_grid]
    for grid in grids:
        for grid_line in grid:
            for idx, grid_dot in enumerate(grid_line):
                if grid_dot not in dot_2_idx.keys():
                    dot_2_idx[grid_dot] = len(dot_2_idx)

    train_grid_idx = [[dot_2_idx[j] for j in i] for i in t_grid]
    valid_grid_idx = [[dot_2_idx[j] for j in i] for i in v_grid]
    infer_grid_idx = [[dot_2_idx[j] for j in i] for i in i_grid]

    train_data = [[str(i), t_grid_label[idx][0]] for idx, i in enumerate(train_grid_idx)]  # 训练轨迹+标签
    valid_data = [[str(i), v_grid_label[idx][0]] for idx, i in enumerate(valid_grid_idx)]  # 验证轨迹+标签
    infer_data = [[str(i), i_grid_label[idx][0]] for idx, i in enumerate(infer_grid_idx)]  # 测试轨迹+标签

    return train_data, valid_data, infer_data, len(dot_2_idx)


if __name__ == "__main__":
    normal_track_dataset = '..\\data\\modify-data\\normal_data.csv'
    abnormal_track_500_dataset = '..\\data\\modify-data\\abnormal_data_500.csv'

    normal_tracks = read_track_csv(normal_track_dataset)
    abnormal_tracks = read_track_csv(abnormal_track_500_dataset)

    tracks_dot_label = track_process(normal_tracks)

    input_dataloader = DataLoader(tracks_dot_label, batch_size=16, shuffle=True,
                                  drop_last=True)

    for batch_data in input_dataloader:
        tracks_dot = strs_2_lists(batch_data[0])
        tracks_label = strs_2_lists(batch_data[1])
        dot_label_pairs = sorted(
            zip(tracks_dot, tracks_label), key=lambda p: len(p[0]), reverse=True)
        tracks_dot_seqs, tracks_label_seqs = zip(*dot_label_pairs)
        tracks_dot_seq, tracks_dot_seq_len = seq_2_tensor(tracks_dot_seqs)
        tracks_label_seq, tracks_label_seq_len = seq_2_tensor(tracks_label_seqs)
