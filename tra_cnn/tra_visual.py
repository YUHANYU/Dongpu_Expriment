r"""
轨迹可视化
    主要是读入train_final.csv数据文件，读取出其中的每一行，
    1、首先对轨迹进行可视化，显示出轨迹的走向、分布密集程度等
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd

start_time = time.time()
data_file = '..\\data\\modify-data\\abnormal_data_1000.csv'
csv_reader = csv.reader(open(data_file))
next(csv_reader)  # 跳过第一行
trajectory_dict = {}  # 将所有轨迹存储在一个字典中
for idx, row in enumerate(csv_reader):
    tra = {}  # 单条轨迹的字典
    tra['trip_id'] = row[0]  # 轨迹id号
    tra['latitude'] = eval(row[1])  # 轨迹纬度数组
    tra['lngitude'] = eval(row[2])  # 轨迹经度数组
    tra['gridline'] = eval(row[3])  # 轨迹网格点
    tra['call_type'] = row[4]  # 出租车呼叫类型
    tra['taxi_id'] = row[5]  # 出租车id
    tra['timestamp'] = row[6]  # 轨迹时间戳
    tra['label'] = row[7]  # 轨迹标签，0为正常轨迹，1为异常轨迹

    trajectory_dict[str(idx)] = tra  # 将每条轨迹存储到轨迹字典中

_1_time = time.time()
take_time = round(_1_time - start_time, 2)
print('读取并建立字典所花费时间', take_time)


def str2list(sample_str):
    """
    将string形式的list转化为list
    :param sample_str: string形式的list
    :return: 正常格式的list
    """
    str_2_list = sample_str.lstrip('[').rstrip(']').split(',')
    str_2_list = [float(i) for i in str_2_list]
    return str_2_list


def grid_line_2_idx(grid_line_dot, long=240, wide=180):
    """
    将轨迹点的序号转化为图片中的行列号
    :param grid_line_dot: 轨迹点
    :param long: 图片长度，默认240
    :param wide: 图片宽度，默认180
    :return: 轨迹点转化后的行列号
    """
    wide_idx = grid_line_dot % 180  # 先对宽度取余，看是否填满这一行
    if wide_idx != 0:  # 没有填满这一行
        long_idx = int(grid_line_dot / wide) + 1  # 行号为填满上一行加1
    else:  # 填满了这一行
        long_idx = int(grid_line_dot / wide)  # 行号为轨迹点直接对宽度求商

    return int(long_idx - 1), int(wide_idx - 1)  # 因为图片矩阵从0开始计数，所以行和列数都减去1


# 先尝试画出热力图
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0, 0)

element_max_num = []
for grid_idx in range(len(trajectory_dict)):
    single_track = trajectory_dict[str(grid_idx)]
    single_track_grid_line = single_track['gridline']
    element_count = pd.value_counts(single_track_grid_line)
    element_max_num.append(max([v for i, v in element_count.items()]))
print(max(element_max_num))

for idx, k in enumerate(range(len(trajectory_dict))):
    _3_time = time.time()
    sample_tra = trajectory_dict[str(k)]  # 取出一条轨迹
    latitude = sample_tra['latitude']  # 经度
    lngitude = sample_tra['lngitude']  # 维度
    grid_line = sample_tra['gridline']  # 轨迹点序列
    tra_label = sample_tra['label']  # 该条轨迹的标签
    tra_id = sample_tra['trip_id']  # 该条轨迹的id号

    assert len(latitude) == len(lngitude) \
           and len(lngitude) == len(grid_line), '经纬度个数和轨迹点个数不一致！'

    map_max_dot = 240 * 180  # 默认的矩阵图片大小

    assert max(grid_line) <= map_max_dot, '此图无法完全显示全部轨迹点！'  # 轨迹点超出图片

    grid_map = np.full((240, 180), 0)  # 底色矩阵图片
    grid_mam_2 = [i for i in range(1, 240 * 180 + 1, 1)]
    grid_mam_assist = np.array([grid_mam_2[j * 180:(j + 1) * 180] for j in range(240)])

    for dot in grid_line:
        if dot == -1:
            continue
        # long_idx, wide_idx = grid_line_2_idx(dot)  # 获取轨迹点在矩阵图片中的行列号
        loc = np.argwhere(grid_mam_assist == dot)
        long_idx, wide_idx = loc[0][0], loc[0][1]
        # grid_map[long_idx][wide_idx] += int(99999 / max(element_max_num))  # 在矩阵图片上，该点数值累积变化
        grid_map[long_idx][wide_idx] += 85  # 推荐在75~95之间

    # grid_line_dict = {}  # 计数每个点出现的次数，存储为字典
    # for idx, dot in enumerate(grid_line):
    #     grid_line_dict[int(dot)] = grid_line_dict.get(dot, 0) + 1
    #
    # for key, value in grid_line_dict.items():
    #     long_idx, wide_idx = grid_line_2_idx(key)
    #     grid_map[long_idx][wide_idx] = value / (240 * 180)

    sns.heatmap(grid_map, cmap='CMRmap_r', xticklabels=0, yticklabels=0, cbar=0,
                vmin=0, vmax=999, linecolor='gainsboro', linewidths=.009, robust=1,
                square=0)  # 显示该轨迹的矩阵图片 gainsboro、whitesmoke
    # TODO 需要对矩阵图片的配色进行修改，使轨迹与底色图片的对比更加明显，
    #  且同时要保证轨迹的分布能从轨迹点的颜色深浅进行区别

    plt.show()
    tra_name = str(tra_id) + "_" + str(int(tra_label[0])) + ".jpg"
    tra_path = '../save/cnn_save\\'
    plt.savefig(tra_path + tra_name, dpi=400)
    plt.clf()  # 重置画布

    _4_time = time.time()
    take_time = round(_4_time - _3_time, 2)
    print('第{}条轨迹耗时{}'.format(idx, take_time))


