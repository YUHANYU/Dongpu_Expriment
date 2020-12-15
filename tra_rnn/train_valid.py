"""
轨迹LSTM模型的训练和验证
"""

import datetime
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

import torch
from torch import optim
import torch.nn.functional as Func
from torch.utils.data import DataLoader

from config import Config

config = Config()

# TODO 将指定目录的路径进行添加
from tqdm import tqdm

from model import TrackLSTM
from data_process import merge_read_csv, track_process, split_train_valid_infer, \
    batch_2_tensor, data_save


class ModelTrainValid():
    def __init__(self, config, model, train_data_loader, valid_data_loader, optimizer):
        self.config = config
        self.model = model
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.optimizer = optimizer

        self.train_all_batch = 0
        self.val_f1_all = [0]
        self.save_checkpoint = 0  # 模型最佳保存点的次数

    def train(self):
        print('\n===开始训练&验证===\n')

        train_log = None
        val_log = None
        if config.log:  # 模型写入日志
            train_log = open(config.train_log, 'w', encoding='utf-8')
            # TODO 待写入模型信息

            train_log.write(
                '轮次：{epoch:3.0f}, 当前批次：{step:3.0f}, 累计批次：{total_step:7.0f}, '
                '批大小：{batch_size:3.0f}, 损失：{loss:10.6f}\n\n'
                    .format(epoch=0, step=0, total_step=0, batch_size=0, loss=0))

            val_log = open(config.valid_log, 'w', encoding='utf-8')
            val_log.write(
                '轮次：{epoch:3.0f}, '
                '批大小：{batch_size:3.0f}, '
                '损失：{loss:10.6f}\n\n, '
                    .format(epoch=0, batch_size=0, loss=0))

        for epoch_idx, epoch in enumerate(range(self.config.epochs)):
            print('[训练轮次 {}]'.format(epoch_idx))
            step = 0  # 每个轮次的批次数
            epoch_start_time = datetime.datetime.now()  # 一个轮次模型的计算开始时间
            each_batch_loss = []
            predict_y_epoch = []
            real_y_epoch = []

            self.model.train()
            self.optimizer.zero_grad()
            for batch_idx, batch_data in enumerate(
                    tqdm(self.train_data_loader, desc='Training...', leave=False)):
                self.train_all_batch += 1
                step += 1

                track_seq, track_seq_len, track_label, _ = batch_2_tensor(batch_data)
                model_out = self.model(track_seq, track_seq_len)
                this_batch_size = model_out.shape[0]
                this_batch_loss = self.__loss_calculate(model_out, track_label)
                this_batch_loss.backward()
                self.optimizer.step()
                each_batch_loss.append(this_batch_loss.detach())  # 获取该批次损失

                test = torch.max(model_out, 1)
                predict_y_batch = torch.max(model_out, 1)[1].data.cpu().numpy()
                predict_y_epoch.extend(predict_y_batch)
                real_y_batch = track_label.view(track_label.size(0)).data.cpu().numpy()
                real_y_epoch.extend(real_y_batch)

                if train_log:
                    train_log.write(
                        '轮次：{epoch:3.0f}, 当前批次：{step:3.0f}, '
                        '累计批次：{total_step:7.0f}, '
                        '批大小：{batch_size:3.0f}, 损失：{loss:10.6f}\n'
                        .format(epoch=epoch_idx, step=step,
                                total_step=self.train_all_batch,
                                batch_size=this_batch_size,
                                loss=this_batch_loss.detach()))

            epoch_end_time = datetime.datetime.now()
            print('批数 %4.0f' % step,
                  '| 累积批数 %8.0f' % self.train_all_batch,
                  '| 批大小 %3.0f' % config.batch_size,
                  '| 耗时', epoch_end_time - epoch_start_time)

            acc, pre, recall, f1_score = self.__metrics(predict_y_epoch, real_y_epoch)
            print('训练',
                  '| 精确度Pre %4.2f' % pre,
                  '| 准确率Acc %4.2f' % acc,
                  '| 召回率Recall %4.2f' % recall,
                  '| F1分数F1-Score %4.2f' % f1_score,
                  '| 首批损失 %10.7f' % each_batch_loss[0],
                  '| 尾批损失 %10.7f' % each_batch_loss[-1])
            if train_log:
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
        :return: 这一批次的损失
        """
        input = pre_label
        target = real_label.view(real_label.size(0))
        loss = Func.cross_entropy(input, target,
                                  ignore_index=config.pad_token,
                                  reduction=config.loss_calculate_mode)

        return loss

    def __valid(self, epoch, val_log, batch):
        self.model.eval()  # 设置模型为验证状态
        predict_y_epoch = []
        real_y_epoch = []

        with torch.no_grad():  # 设置验证产生的损失不更新模型
            for _, batch_data in enumerate(tqdm(
                    self.valid_data_loader, desc='Validating...', leave=False)):
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
            print('验证',
                  '| 精确度Pre %4.2f' % pre,
                  '| 准确率Acc %4.2f' % acc,
                  '| 找回率Recall %4.2f' % recall,
                  '| F1分数F1-Score %4.2f\n' % f1_score)

        if config.save_trained_model:
            model_state_dict = self.model.state_dict()  # 保存训练模型状态
            checkpoint = {  # 保存点的信息
                'model': model_state_dict,
                'settings': config,
                'epoch': epoch,
                'total batch': batch}

            if config.save_trained_model_type == 'ALL':
                model_name = config.sava_base_path + 'all_f1-{f1:4.2f}.chkpt'. \
                    format(f1=f1_score)
                torch.save(checkpoint, model_name)
            elif config.save_trained_model_type == 'BEST':
                model_name = config.sava_base_path + 'best_f1-{f1:4.2f}.chkpt'. \
                    format(f1=f1_score)
                if f1_score >= max(self.val_f1_all):
                    torch.save(checkpoint, model_name)
                    self.save_checkpoint += 1
                    print('已经第{}次更新模型最佳保存点！'.format(self.save_checkpoint))

    def __metrics(self, y_pred, y_ture):
        accuracy = round(accuracy_score(y_ture, y_pred), 2)
        precision = round(precision_score(y_ture, y_pred), 2)
        recall = round(recall_score(y_ture, y_pred), 2)
        f1 = round(f1_score(y_ture, y_pred), 2)

        return accuracy, precision, recall, f1


if __name__ == "__main__":
    tracks = merge_read_csv(config.abnormal_tra_1500_dataset, config.normal_tra_dataset)

    tracks, max_track_dot = track_process(tracks)
    train_track, valid_track, infer_track = split_train_valid_infer(tracks, config)
    data_save(train_track, 'train')  # 训练轨迹数据保存
    data_save(valid_track, 'valid')  # 验证轨迹数据保存
    data_save(infer_track, 'infer')  # 推理轨迹数据保存

    train_track_data_loader = DataLoader(train_track, batch_size=config.batch_size,
                                         shuffle=True, drop_last=False)
    valid_track_data_loader = DataLoader(valid_track, batch_size=16,
                                         shuffle=True, drop_last=False)

    track_lstm_model = TrackLSTM(  # 轨迹LSTM
        input_size=max_track_dot,
        hidden_size=64,
        bi_lstm=config.bi_lstm,
        n_layers=config.n_layers,
        dropout=config.dropout).to(config.device)

    optimizer = optim.Adam(track_lstm_model.parameters(), lr=config.lr)

    model_train_valid = ModelTrainValid(
        config=config,
        model=track_lstm_model,
        train_data_loader=train_track_data_loader,
        valid_data_loader=valid_track_data_loader,
        optimizer=optimizer)
    model_train_valid.train()
