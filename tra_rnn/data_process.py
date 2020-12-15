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
    tracks = {}  # 所有轨迹的字典
    for idx, row in enumerate(csv_reader):
        track = {}  # 单条轨迹字典

        track['trip_id'] = int(row[0])  # 轨迹id号
        track['latitude'] = eval(row[1])  # 轨迹纬度数组
        track['lngitude'] = eval(row[2])  # 轨迹经度数组
        row_3_list = str_2_list(row[3])
        if -1 not in row_3_list:
            track['gridline'] = eval(row[3])  # 轨迹网格点
        else:
            continue
        track['call_type'] = row[4]  # 出租车呼叫类型
        track['taxi_id'] = row[5]  # 出租车id
        track['timestamp'] = row[6]  # 轨迹时间戳
        track['label'] = row[7]

        # if int(row[7]) == 1:  # 异常轨迹
        #     track['label'] = str([0, 1])
        # else:  # 正常轨迹
        #     track['label'] = str([1, 0])

        tracks[str(idx)] = track  # 将每条轨迹存储到轨迹集合字典中

    return tracks


def str_2_list(tgt_seq):
    """
    转化字符序列为列表序列，用于转化批数据中的字符序列
    :param tgt_seq: 输入的字符序列
    :return: 字符序列对应的列表序列
    """
    s = tgt_seq.lstrip('[').rstrip(']').split(',')
    ss = [int(i) for i in s]

    return ss


def strs_2_lists(tgt_seq):
    """
    转化字符序列为列表序列，用于转化批数据中的字符序列
    :param tgt_seq: 输入的字符序列
    :return: 字符序列对应的列表序列
    """
    ss = []
    for s in tgt_seq:  # 把字符序列还原list序列
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
                     for line in tracks_grid_dot]

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
    tracks_dot = strs_2_lists(batch_data[0])
    tracks_label = strs_2_lists(batch_data[1])
    dot_label_pairs = sorted(
        zip(tracks_dot, tracks_label), key=lambda p: len(p[0]), reverse=True)
    tracks_dot_seqs, tracks_label_seqs = zip(*dot_label_pairs)
    tracks_dot_seq, tracks_dot_seq_len = seq_2_tensor(tracks_dot_seqs)
    tracks_label_seq, tracks_label_seq_len = seq_2_tensor(tracks_label_seqs)

    return tracks_dot_seq, tracks_dot_seq_len, tracks_label_seq, tracks_label_seq_len


def split_train_valid_infer(tracks, config):
    """

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


def data_save(original_data, data_type='train'):
    """
    将轨迹数据以字典形式保存下来
    :param original_data: 原数据数据，list格式
    :param data_type: 保存的数据类型，默认为训练
    :return:
    """
    track_len = len(original_data)
    data_dict = {
        'data_length': str(track_len),
        'data_contend': original_data,
    }
    data_name = config.sava_base_path + data_type + '.pt'
    torch.save(data_dict, data_name)



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
