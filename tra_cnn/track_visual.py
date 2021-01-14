r"""
将轨迹点转化成为可视化图片，以一个函数形式向外提供功能
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm
import os


def str_2_list(sample_str):
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
        tra['id'] = row[1]  # 轨迹id号
        tra['latitude'] = eval(row[2])  # 轨迹纬度数组
        tra['longitude'] = eval(row[3])  # 轨迹经度数组
        tra['grid_point'] = eval(row[4])  # 轨迹网格点
        tra['call_type'] = row[5]  # 出租车呼叫类型
        tra['taxi_id'] = row[6]  # 出租车id
        tra['timestamp'] = row[7]  # 轨迹时间戳
        tra['label'] = row[8]  # 轨迹标签，0为正常轨迹，1为异常轨迹

        track_dict[str(idx)] = tra  # 将每条轨迹存储到轨迹字典中

    return track_dict


def track_2_picture(data_set, save_path, length, width, show_line=0, display_pic=0):
    """
    将轨迹转化为图片，并保存在本地指定路径
    :param data_set: 轨迹数据集
    :param save_path: 图片保存路径
    :param length: 图片长度
    :param width: 图片宽度
    :param show_line: 是否展示图片网格线
    :param display_pic: 展示图片，默认为否
    :return:
    """
    start_time = time.time()

    # 热力图底图
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    track_dict = read_csv(data_set)  # 读取轨迹数据集形成轨迹字典

    error_grid_point = 0  # 错误轨迹点条数
    for idx in tqdm(range(len(track_dict)), leave=False, desc='Processing...'):
        single_track_dict = track_dict[str(idx)]  # 取出单条轨迹
        latitude = single_track_dict['latitude']  # 轨迹经度
        longitude = single_track_dict['longitude']  # 轨迹纬度
        grid_point = single_track_dict['grid_point']  # 轨迹点
        track_label = single_track_dict['label']  # 轨迹标签
        track_id = single_track_dict['id']  # 轨迹id

        assert len(latitude) == len(longitude) \
            and len(longitude) == len(grid_point), '经度、纬度、轨迹点三者长度不一致！'

        max_point = length * width  # 图片网格点最大数
        assert max(grid_point) <= max_point, '轨迹点超出图片最大限制！'

        track_picture = np.full((length, width), 0)  # 绘制白色底图
        grid_point_list = [i for i in range(1, max_point+1, 1)]  # 轨迹点数组
        grid_point_array = np.array([grid_point_list[j*width:(j+1)*width]  # 轨迹点矩阵点
                                     for j in range(length)])  # 每个元素==图片像素点，并给像素点标序号

        for point in grid_point:  # 轨迹点转化
            if point == -1:  # 排除异常轨迹
                error_grid_point += 1  # 记录异常轨迹的数量
                continue
            location = np.argwhere(grid_point_array == point)  # 在轨迹矩阵中定位轨迹点
            length_idx, width_idx = location[0][0], location[0][1]  # 获取轨迹点坐标
            track_picture[length_idx][width_idx] += 85  # 轨迹点数值变大，对应颜色加深

        ax = sns.heatmap(track_picture, cmap='CMRmap_r', xticklabels=0, yticklabels=0, cbar=0, vmin=0,
                    vmax=999, linewidths=0.009,
                    linecolor='gainsboro' if show_line else 'white')  # 轨迹热力图
        heatmap = ax.get_figure()
        pic_name = str(round(float(track_id), 5)) + "_" + \
                   str(int(track_label[0])) + '.jpg'  # 轨迹图片命名
        heatmap.savefig(save_path + pic_name, dpi=800)  # 轨迹图片保存
        if display_pic:  # 是否展示图片
            plt.show()
        plt.clf()  # 重置画布

    end_time = time.time()
    take_time = round(end_time - start_time, 2)
    picture_num = idx - error_grid_point
    print('耗时{}s处理完成！保存{}张图片，出现{}条错误轨迹，已排除，请检查！'.
          format(take_time, picture_num, error_grid_point))


if __name__ == "__main__":
    run_parm = {'Grid200':{'length': 240, 'width': 180},
                'Grid300':{'length': 165, 'width': 120},
                'Grid400':{'length': 124, 'width': 90}}
    base_path = '..\\data\\'
    grids = ['Grid200\\', 'Grid300\\', 'Grid400\\']
    save_base_path = '..\\save\\cnn_save\\'
    for grid in grids:
        files = os.listdir(base_path + grid)
        for f in files:
            save_path = save_base_path + grid + f.split('.')[0] + '\\'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            track_2_picture(data_set=base_path + grid + f,
                            save_path= save_path,
                            length=run_parm[grid.split('\\')[0]]['length'],
                            width=run_parm[grid.split('\\')[0]]['width'])