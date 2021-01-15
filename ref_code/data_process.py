r"""
处理轨迹数据形成ATD-RNN模型能输入的格式
"""
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

def write_txt(base_path, dataset, save_path):
    """
    将数据集写成指定格式的TXT文件
    :param dataset: 数据集
    :param save_path: 保持路径
    :return: 文件处理时间
    """
    track_dict = read_csv(base_path + dataset)
    file = save_path + dataset.split('.csv')[0] + '.txt'
    with open(file, 'w', encoding='utf-8') as f:
        for key, value in track_dict.items():
            label = value['label']
            timestamp = value['timestamp']
            grid_point = value['grid_point']
            if -1 in grid_point:
                continue
            f.write(label + ' ' + timestamp + ' ')
            for i in grid_point:
                f.write(str(i) + ' ')
            f.write('\n')


if __name__ == "__main__":
    run_parm = {'Grid200':{'length': 240, 'width': 180},
                'Grid300':{'length': 165, 'width': 120},
                'Grid400':{'length': 124, 'width': 90}}
    base_path = '..\\data\\'
    grids = ['Grid200\\', 'Grid300\\', 'Grid400\\']
    save_base_path = '..\\save\\ref_code_save\\'
    for grid in grids:
        files = os.listdir(base_path + grid)
        for f in files:
            save_path = save_base_path + grid + f.split('.')[0] + '\\'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            write_txt(base_path=base_path + grid, dataset=f, save_path= save_path)
