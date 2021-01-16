r"""
ATD-RNN模型训练和验证
"""

import copy
import random
import numpy as np
import tensorflow as tf
from sklearn import metrics
import os


class Config(object):
    """参数类"""

    embedding_dim = 128  # 嵌入层大小
    seq_length = 1000  # 轨迹点最大长度
    num_classes = 2  # 轨迹类别
    vocab_size = 5000  #轨迹点最大个数

    num_layers = 4  # RNN层数
    hidden_dim = 128  # RNN隐藏层大小
    rnn = 'lstm'  # RNN类型

    dropout_keep_prob = 0.5  # dropout大小
    learning_rate = 1e-4  # 损失函数学习率

    batch_size = 32  # 批大小
    num_epochs = 40  # 训练轮次


class RNN(object):
    """ATD-RNN核心计算模型"""
    def __init__(self, config):
        self.config = config  # 配置类对象

        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length])  # 模型x输入，None*轨迹线最大值
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes])  # 模型y输入，None*轨迹类别
        self.keep_prob = tf.placeholder(tf.float32, [], name="keep_prob")  # dropout层，[]
        self.seq_length = tf.placeholder(tf.int32, [None])  # 序列长度，[None]
        self.rnn()

    def rnn(self):

        def lstm_cell():
            return tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_dim,
                                                forget_bias=1.0, state_is_tuple=True)

        def gru_cell():
            return tf.contrib.rnn.GRUCell(self.config.hidden_dim)

        def dropout():
            if (self.config.rnn == "lstm"):
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        # with tf.device('/gpu:0'):  # 选择GPU还是CPU跑模型
        with tf.device('cpu'):
            embedding = tf.get_variable("embedding", [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("rnn"):
            cells = [dropout() for _ in range(self.config.num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

            embedding_inputs = tf.transpose(embedding_inputs, [1, 0, 2])
            _outputs, self._states = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs,
                                                       sequence_length=self.seq_length, time_major=True,
                                                       dtype=tf.float32)
            _outputs = tf.transpose(_outputs, [1, 0, 2])
            # last = _outputs[:,-1,:]
            print(self._states)
            last = tf.concat([item.c for item in self._states], axis=1)

        with tf.name_scope("score"):
            fc = tf.layers.dense(last, self.config.hidden_dim, name='fc1')
            # fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.sigmoid(fc)

            self.logits = tf.layers.dense(fc, self.config.num_classes, name="fc2")
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)

        with tf.name_scope("optimize"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            self.optim = tf.train.AdamOptimizer(
                learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def load_data(train_dir, test_dir):
    """
    加载数据
    :param train_dir: 训练数据集
    :param test_dir: 测试数据集
    :return:
    """
    # train_dir = "..\\data\\atd-rnn\\Train_2_3_4.txt"  # 训练数据集
    # test_dir = "..\\data\\atd-rnn\\Test_2_3_4.txt"  # 验证数据集
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    with open(train_dir, "r") as fin:
        for line in fin:
            tmp = line.strip().split(" ")
            train_data.append(tmp[2:])
            if tmp[0] == '1':
                train_labels.append([0, 1])
                # 负样本重复
                for _ in range(2):
                    item = copy.deepcopy(tmp[2:])
                    for j, k in enumerate(item):
                        # 0.6 的概率产生扰动
                        if random.random() < 0.6:
                            continue
                        item[j] = str(int(k) + (1 if random.random() < 0.5 else -1) * int(
                            random.random() * 100))  # 正负方向上随机移动100个位置
                        while (int(item[j]) < 0):
                            item[j] = str(int(k) + (1 if random.random() < 0.5 else - 1) * int(
                                random.random() * 100))  # 移动到小于0 的时候重新移动
                    train_data.append(item)
                    train_labels.append(copy.deepcopy(train_labels[-1]))
            else:
                train_labels.append([1, 0])
    with open(test_dir, "r") as fin:
        for line in fin:
            tmp = line.strip().split(" ")
            test_data.append(tmp[2:])
            if (tmp[0] == '1'):
                test_labels.append([0, 1])
            else:
                test_labels.append([1, 0])
    word2id = {}
    id2word = {}
    for line in train_data:
        for item in line:
            if item not in word2id:
                word2id[item] = len(word2id)
    for line in test_data:
        for item in line:
            if item not in word2id:
                word2id[item] = len(word2id)
    id2word = {value: key for key, value in word2id.items()}
    train_seq_length = list(map(len, train_data))
    max_length1 = max(train_seq_length)
    test_seq_length = list(map(len, test_data))
    max_length2 = max(test_seq_length)
    max_length = max(max_length1, max_length2)
    for i, line in enumerate(train_data):
        for j, item in enumerate(line):
            train_data[i][j] = word2id[item]
    for i, line in enumerate(test_data):
        for j, item in enumerate(line):
            test_data[i][j] = word2id[item]
    # padding
    for i, line in enumerate(train_data):
        if (len(line) < max_length):
            line.extend([0] * (max_length - len(line)))
    for i, line in enumerate(test_data):
        if (len(line) < max_length):
            line.extend([0] * (max_length - len(line)))

    return train_data, train_labels, train_seq_length, \
           test_data, test_labels, test_seq_length, \
           max_length, word2id, id2word


def load_test_data(filename):
    """
    加载测试数据
    :param filename:  文件名
    :return:
    """
    res_data = []
    res_labels = []
    res_seq_length = []
    with open(filename, "r") as fin:
        for line in fin:
            tmp = line.strip().split()
            if (tmp[0] == "0"):
                res_labels.append([1, 0])
            else:
                res_labels.append([0, 1])
            tmp = tmp[2:]
            n = len(tmp)
            res_seq_length.append(n)
            tmp_data = [word2id[item] for item in tmp]
            tmp_data = tmp_data + [0] * (max_length - n)
            res_data.append(tmp_data)
    return res_data, res_labels, res_seq_length


def generate_batch(train_data, train_labels, train_seq_length, batch_size):
    """
    生成批数据
    :param train_data: 训练数据
    :param train_labels: 训练数据的标签
    :param train_seq_length: 训练数据的长度
    :param batch_size: 批大小
    :return:
    """
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    seq_length = np.array(train_seq_length)
    print("train_data.shape=", train_data.shape)
    print("train_labels.shape=", train_labels.shape)
    indics = np.random.permutation(len(train_data))
    train_data = train_data[indics]
    train_labels = train_labels[indics]
    seq_length = seq_length[indics]
    train_len = len(train_data)
    num_iters = train_len // batch_size
    if (num_iters * batch_size != train_len):
        num_iters += 1
    for i in range(num_iters):
        start = i * batch_size
        end = min(start + batch_size, train_len)
        yield train_data[start: end].tolist(), train_labels[start: end].tolist(), seq_length[start: end].tolist()


def train(train_data, train_label, train_seq_length, model):
    acc_value = []
    recall_value = []
    precision_value = []
    f1_value = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        acc = 0
        allAcc = 0
        for i in range(model.config.num_epochs):
            for batch_x, batch_y, batch_length in generate_batch(train_data, train_label, train_seq_length,
                                                                 model.config.batch_size):

                _states, _loss, _ = sess.run([model._states, model.loss, model.optim], feed_dict={
                    model.input_x: batch_x,
                    model.input_y: batch_y,
                    model.keep_prob: model.config.dropout_keep_prob,
                    model.seq_length: batch_length
                })

                _predict, _acc = sess.run([model.y_pred_cls, model.acc], feed_dict={
                    model.input_x: batch_x,
                    model.input_y: batch_y,
                    model.keep_prob: model.config.dropout_keep_prob,
                    model.seq_length: batch_length
                })

                _test_acc = sess.run(model.acc, feed_dict={
                    model.input_x: test_data,
                    model.input_y: test_labels,
                    model.keep_prob: model.config.dropout_keep_prob,
                    model.seq_length: test_seq_length
                })

                tmp = [_test_acc]
                tmp_precision = [0]
                tmp_recall = [0]
                tmp_f1 = [0]
                for filename in testDirs:
                    tmp_data, tmp_labels, tmp_seq = load_test_data(filename)
                    predict = sess.run(model.y_pred_cls, feed_dict={
                        model.input_x: tmp_data,
                        model.input_y: tmp_labels,
                        model.keep_prob: model.config.dropout_keep_prob,
                        model.seq_length: tmp_seq
                    })
                    # print(predict)
                    tmp_labels = [0 if item[0] == 1 else 1 for item in tmp_labels]
                    acc = metrics.accuracy_score(tmp_labels, predict)
                    precision = metrics.precision_score(tmp_labels, predict)
                    recall = metrics.recall_score(tmp_labels, predict)
                    f1 = metrics.f1_score(tmp_labels, predict)
                    tmp.append(acc)
                    tmp_precision.append(precision)
                    tmp_recall.append(recall)
                    tmp_f1.append(f1)
                acc_value.append(tmp)
                precision_value.append(tmp_precision)
                recall_value.append(tmp_recall)
                f1_value.append(tmp_f1)
                print(
                    "at step %d loss=%.6f \t acc=%.6f \t test_acc = %.6f\n _acc=%s\n precisions=%s \n recalls=%s \n f1=%s" % (
                        step, _loss, _acc, _test_acc, tmp, tmp_precision, tmp_recall, tmp_f1))
                step += 1

    return acc_value, precision_value, recall_value, f1_value


if __name__ == "__main__":
    ####### 训练过程 ##########
    train_dir = None
    test_dir = None
    train_data, train_labels, train_seq_length, \
    test_data, test_labels, test_seq_length, \
    max_length, word2id, id2word = load_data(train_dir, test_dir)
    n = len(train_data)
    index = np.random.permutation(n)
    train_data = np.array(train_data)
    train_data = train_data[index].tolist()
    train_labels = np.array(train_labels)
    train_labels = train_labels[index].tolist()
    train_seq_length = np.array(train_seq_length)
    train_seq_length = train_seq_length[index].tolist()

    print("训练集中异常轨迹的比例: {}".  # TODO 这里要搞清楚比例是怎么算出来的
          format(sum([item[1] for item in train_labels]) / len(train_labels)))
    print("测试集中异常轨迹的比例: {}".  # TODO 这里要搞清楚比例是怎么算出来的
          format(sum(item[1] for item in test_labels) / len(test_labels)))

    config = Config()
    config.seq_length = max_length
    config.vocab_size = len(word2id)
    model = RNN(config)




