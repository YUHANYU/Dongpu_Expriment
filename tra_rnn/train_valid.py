"""
轨迹LSTM模型的训练和验证
"""

import os
import datetime
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

import torch
from torch import nn
from torch import optim
import torch.nn.functional as Func
from torch.utils.data import DataLoader

from config import Config

config = Config()

# TODO 将指定目录的路径进行添加
from tqdm import tqdm

from model import TrackLSTM
from data_process import merge_read_csv, track_process, split_train_valid_infer, \
    batch_2_tensor, data_save, read_track_csv, load_data, track_merge_line_label


class ModelTrainValid():
    def __init__(self, config, model, train_data_loader, valid_data_loader, optimizer, grid, percent):
        """
        TrackRNN模型训练和验证
        :param config: 参数对象
        :param model: 计算模型
        :param train_data_loader: 训练数据批量加载器
        :param valid_data_loader: 验证数据批量加载器
        :param optimizer: 优化器
        """
        self.config = config
        self.model = model
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.optimizer = optimizer
        self.grid = grid
        self.percent = percent

        self.train_all_batch = 0
        self.val_f1_all = [0]
        self.save_checkpoint = 0  # 模型最佳保存点的次数

    def train(self):
        print('\n===开始训练&验证===\n')

        train_log, val_log = None, None
        if config.log:  # 模型写入日志
            train_log = open(config.sava_base_path + self.grid + self.percent+ '\\' + 'train_log.txt',
                             'w', encoding='utf-8')  # 训练日志
            train_log.write(
                '轮次：{epoch:3.0f}, 当前批次：{step:3.0f}, 累计批次：{total_step:7.0f}, '
                '批大小：{batch_size:3.0f}, 损失：{loss:10.6f}, 学习率：{lr:10.6f}\n\n'
                    .format(epoch=0, step=0, total_step=0, batch_size=0, loss=0, lr=0))

            val_log = open(config.sava_base_path + self.grid + self.percent+ '\\' + 'valid_log.txt',
                           'w', encoding='utf-8')  # 验证日志
            val_log.write(
                '轮次：{epoch:3.0f}, '
                '批大小：{batch_size:3.0f}, '
                '损失：{loss:10.6f}\n\n, '
                    .format(epoch=0, batch_size=0, loss=0))

        for epoch_idx, epoch in enumerate(range(self.config.epochs)):  # 轮次循环
            print('[训练轮次 {}]'.format(epoch_idx))
            step = 0  # 每个轮次的批次数
            epoch_start_time = datetime.datetime.now()  # 一个轮次模型的计算开始时间
            each_batch_loss = []
            predict_y_epoch = []
            real_y_epoch = []

            self.model.train()
            for batch_idx, batch_data in enumerate(self.train_data_loader):  # 批次循环
                self.optimizer.zero_grad()  # 轮次训练前梯度清零
                self.train_all_batch += 1  # 训练批次数只加1
                step += 1  # 训练步数

                track_seq, track_seq_len, track_label, _ = batch_2_tensor(batch_data)  # 批次数据展开转换
                model_out = self.model(track_seq, track_seq_len)  # 模型计算输出
                this_batch_size = model_out.shape[0]  # 批大小
                this_batch_loss = self.__loss_calculate(model_out, track_label)  # 批平均损失
                this_batch_loss.backward()  # 损失反向传播
                self.optimizer.step()  # 优化器步进
                print(this_batch_loss)
                each_batch_loss.append(this_batch_loss.item())  # 获取该批次损失

                predict_y_batch = torch.max(model_out, 1)[1].data.cpu().numpy()  # 当前批次预测结果
                predict_y_epoch.extend(predict_y_batch)
                real_y_batch = track_label.view(track_label.size(0)).data.cpu().numpy()  # 当前批次实际结果
                real_y_epoch.extend(real_y_batch)


                if config.log:  # 训练日志
                    train_log.write(
                        '轮次：{epoch:3.0f}, 当前批次：{step:3.0f}, '
                        '累计批次：{total_step:7.0f}, '
                        '批大小：{batch_size:3.0f}, 损失：{loss:10.6f}\n'
                        .format(epoch=epoch_idx, step=step,
                                total_step=self.train_all_batch,
                                batch_size=this_batch_size,
                                loss=this_batch_loss.item()))

            epoch_end_time = datetime.datetime.now()  # 一轮次结束时间点
            print('批数 %4.0f' % step,
                  '| 累积批数 %8.0f' % self.train_all_batch,
                  '| 批大小 %3.0f' % config.batch_size,
                  '| 耗时', epoch_end_time - epoch_start_time)

            acc, pre, recall, f1_score = self.__metrics(predict_y_epoch, real_y_epoch)  # 计算当前轮次训练后的模型指标
            print('训练',
                  '| 精确度Pre %4.2f' % pre,
                  '| 准确率Acc %4.2f' % acc,
                  '| 召回率Recall %4.2f' % recall,
                  '| F1分数F1-Score %4.2f' % f1_score,
                  '| 首批损失 %10.7f' % each_batch_loss[0],
                  '| 尾批损失 %10.7f' % each_batch_loss[-1])

            if config.log:
                train_log.write(
                    '精确度{pre:6.3f} | 准确率acc{acc:6.3f} | 召回率Recall{recall:6.3f} '
                    '| F1分数F1-Score{f1_score:6.3f}\n\n'.
                        format(pre=pre, acc=acc, recall=recall, f1_score=f1_score))

            # TODO 记录并打印训练情况
            self.__valid(epoch_idx, val_log, self.train_all_batch)  # 每个轮次结束后验证一次

        train_log.close()
        val_log.close()

    def __loss_calculate(self, pre_label, real_label):
        """
        计算模型的损失
        :param pre_label: 预测的标签
        :param real_label: 真实的标签
        :return: 该批次的损失
        """
        input = pre_label  # 模型输入 [batch_size, num_classes]
        target = real_label.view(real_label.size(0))  # 真实标签 [num_classes]
        loss = torch.nn.functional.cross_entropy(input, target, reduction=config.loss_calculate_mode)

        return loss

    def __valid(self, epoch, val_log, batch):
        self.model.eval()  # 设置模型为验证状态
        predict_y_epoch = []
        real_y_epoch = []

        with torch.no_grad():  # 设置验证产生的损失不更新模型
            for _, batch_data in enumerate(self.valid_data_loader):
                track_seq, track_seq_len, track_label, _ = batch_2_tensor(batch_data)
                model_out = self.model(track_seq, track_seq_len)
                this_batch_size = model_out.shape[0]
                this_batch_loss = self.__loss_calculate(model_out, track_label)

                predict_y_batch = torch.max(model_out, 1)[1].data.cpu().numpy()
                predict_y_epoch.extend(predict_y_batch)
                real_y_batch = track_label.view(track_label.size(0)).data.cpu().numpy()
                real_y_epoch.extend(real_y_batch)

                if val_log:
                    val_log.write(
                        '轮次：{epoch:3.0f}, 批大小：{batch_size:3.0f}, 损失：{loss:10.7f}\n\n'
                            .format(epoch=epoch, batch_size=this_batch_size,
                                    loss=this_batch_loss.detach()))

            acc, pre, recall, f1_score = self.__metrics(predict_y_epoch, real_y_epoch)
            self.val_f1_all.append(f1_score)
            print('验证',
                  '| 精确度Pre %4.2f' % pre,
                  '| 准确率Acc %4.2f' % acc,
                  '| 找回率Recall %4.2f' % recall,
                  '| F1分数F1-Score %4.2f' % f1_score)

        if config.save_trained_model:
            model_state_dict = self.model.state_dict()  # 保存训练模型状态
            checkpoint = {  # 保存点的信息
                'model': model_state_dict,
                'settings': config,
                'epoch': epoch,
                'total batch': batch}

            if config.save_trained_model_type == 'ALL':
                model_name = config.sava_base_path + self.grid + self.percent+ '\\' + 'all_f1-{f1:4.2f}.chkpt'. \
                    format(f1=f1_score)
                torch.save(checkpoint, model_name)
            elif config.save_trained_model_type == 'BEST':
                model_name = config.sava_base_path + self.grid + self.percent+ '\\' + 'best_f1-{f1:4.2f}.chkpt'. \
                    format(f1=f1_score)
                if f1_score >= max(self.val_f1_all):
                    torch.save(checkpoint, model_name)
                    self.save_checkpoint += 1
                    print('已经第{}次更新模型最佳保存点！\n'.format(self.save_checkpoint))

    def __metrics(self, y_pred, y_ture):
        accuracy = round(accuracy_score(y_ture, y_pred), 3)
        precision = round(precision_score(y_ture, y_pred), 3)
        recall = round(recall_score(y_ture, y_pred), 3)
        f1 = round(f1_score(y_ture, y_pred), 3)

        return accuracy, precision, recall, f1


if __name__ == "__main__":
    grids = ['Grid300\\', 'Grid400\\']  # 数据集类型
    for grid in grids:
        if not os.path.exists(config.sava_base_path + grid):  # 判定是否有特定数据集的保存路径
            os.makedirs(config.sava_base_path + grid)

        files = os.listdir(config.data_base_path + grid)  # 特定数据集下所有文件
        files.remove('valid.csv')  # 排除验证数据文件
        files.remove('test.csv')  # 排除推理数据文件
        for file in files:
            train_file = config.data_base_path + grid + file  # 训练数据文件
            valid_file = config.data_base_path + grid + 'valid.csv'  # 验证数据文件
            infer_file = config.data_base_path + grid + 'test.csv'  # 测试数据文件

            percent = 'baseline' if 'base' in file else file.split('.csv')[0].split('train_')[1]
            if not os.path.exists(config.sava_base_path + grid + percent):
                percent_dir = config.sava_base_path + grid + percent
                os.makedirs(percent_dir)

            train_track = track_merge_line_label(read_track_csv(train_file))  # 训练轨迹
            valid_track = track_merge_line_label(read_track_csv(valid_file))  # 验证轨迹
            infer_track = track_merge_line_label(read_track_csv(infer_file))  # 测试轨迹

            train_data, valid_data, infer_data, dot_max_len = load_data(
                train_track, valid_track, infer_track)  # 加载训练、验证和测试数据

            data_save(train_data, grid, percent, 'train')  # 训练轨迹数据保存
            data_save(valid_data, grid, percent, 'valid')  # 验证轨迹数据保存
            data_save(infer_data, grid, percent, 'infer')  # 推理轨迹数据保存

            train_data_loader = DataLoader(train_data, batch_size=config.batch_size,
                                           shuffle=True, drop_last=False)  # 训练数据加载
            valid_data_loader = DataLoader(valid_data, batch_size=config.batch_size,
                                           shuffle=True, drop_last=False)  # 验证数据加载
            infer_data_loader = DataLoader(infer_data, batch_size=config.batch_size,
                                           shuffle=True, drop_last=False)  # 测试数据加载

            track_lstm_model = TrackLSTM(  # 轨迹LSTM模型
                input_size=dot_max_len,
                hidden_size=config.hidden_size,
                bi_lstm=config.bi_lstm,
                n_layers=config.n_layers,
                dropout=config.dropout).to(config.device)

            # TODO 尝试使用Transformer的encoder部分作为轨迹特征提取器

            optimizer = optim.SGD(track_lstm_model.parameters(), lr=config.lr)  # 模型优化器

            model_train_valid = ModelTrainValid(  # 模型训练-验证
                config=config,
                model=track_lstm_model,
                train_data_loader=train_data_loader,
                valid_data_loader=valid_data_loader,
                optimizer=optimizer,
                grid=grid,
                percent=percent)
            model_train_valid.train()  # 模型训练，自带验证

            # TODO 模型推理

