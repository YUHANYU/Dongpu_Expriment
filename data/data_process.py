r"""处理数据集，删掉数据集中gridline出现过-1的行"""

import csv
import os


def read_csv(file):
    """
    读取轨迹数据集，并且返回轨迹数据字典
    :param file: 轨迹数据集
    :return: 轨迹数据字典
    """
    csv_reader = csv.reader(open(file))  # 打开读取csv数据文件
    next(csv_reader)  # 跳过第一行
    track_dict = {}  # 空轨迹字典
    for idx, row in enumerate(csv_reader):
        tra = {}  # 单条轨迹的字典
        tra['id'] = row[0]  # 轨迹id号
        tra['latitude'] = eval(row[1])  # 轨迹纬度数组
        tra['longitude'] = eval(row[2])  # 轨迹经度数组
        tra['grid_point'] = eval(row[3])  # 轨迹网格点
        tra['call_type'] = row[4]  # 出租车呼叫类型
        tra['taxi_id'] = row[5]  # 出租车id
        tra['timestamp'] = row[6]  # 轨迹时间戳
        tra['label'] = row[7]  # 轨迹标签，0为正常轨迹，1为异常轨迹

        track_dict[str(idx)] = tra  # 将每条轨迹存储到轨迹字典中

    return track_dict


def read_process_write_csv(read_file, write_file):
    """
    读取csv文件，并且处理csv文件中gridline含有-1的行，再写回原文件中
    :param file:目标文件
    :return:
    """
    with open(read_file, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)  # 打开读取csv数据文件
        next(csv_reader)  # 跳过第一行
        track_dict = {}  # 空轨迹字典
        for idx, row in enumerate(csv_reader):
            tra = {}  # 单条轨迹的字典
            tra['id'] = row[0]  # 轨迹id号
            tra['latitude'] = eval(row[1])  # 轨迹纬度数组
            tra['longitude'] = eval(row[2])  # 轨迹经度数组
            tra['grid_point'] = eval(row[3])  # 轨迹网格点
            if -1 in tra['grid_point']:
                print('有一条！')
                continue
            tra['call_type'] = row[4]  # 出租车呼叫类型
            tra['taxi_id'] = row[5]  # 出租车id
            tra['timestamp'] = row[6]  # 轨迹时间戳
            tra['label'] = row[7]  # 轨迹标签，0为正常轨迹，1为异常轨迹

            track_dict[str(idx)] = tra  # 将每条轨迹存储到轨迹字典中

    with open(write_file, 'w', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["trip_id", "latitude", "lngitude", "gridline", "call_type",
                             "taxi_id", "timestamp", "labels"])
        for key, value in track_dict.items():
            csv_writer.writerow(value)


if __name__ == "__main__":
    base_path = '.\\'
    grids = ['Grid200\\', 'Grid300\\', 'Grid400\\']
    grids_new = ['Grid200_no_-1\\', 'Grid300_no_-1\\', 'Grid400_no_-1\\']
    for idx in range(len(grids)):
        new_path = base_path + grids_new[idx]
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        files = os.listdir(base_path + grids[idx])
        for f in files:
            read_process_write_csv(read_file=base_path + grids[idx] + f,
                                   write_file=base_path + grids_new[idx] + f)
