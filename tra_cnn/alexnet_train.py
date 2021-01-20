
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from alexnet import AlexNet
import os
import json
import torchvision.models as models
#config
batch_size=32
EPOCH=100
#文件路径
image_path='../save/cnn_save/Grid200/abnormal_5percent_pic'


# 如果当前有可以使用的GPU设备的话，就使用第一块GPU，否则使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = {
    # 当key为train时，训练集预处理方法  ；  随即裁剪224*224
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),  # 水平上随机翻转
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}


# datasets.ImageFolder加载数据集，data_transform数据预处理
train_dataset = datasets.ImageFolder(root=image_path + "/train",
                                     transform=data_transform["train"])
train_num = len(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)

validate_dataset = datasets.ImageFolder(root=image_path + "/valid",
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=True,
                                              num_workers=0)

# #模型
# net = models.alexnet(pretrained=True)      # 加载预训练的模型，原来的模型是6分类
#
# net.classifier[6]==nn.Linear(4096,2)

net=AlexNet(num_classes=2)
save_path='./models/Alexnet.pth'
# 网络指定到规定的设备中
net.to(device)
loss_function = nn.CrossEntropyLoss()
# pata = list(net.parameters());这个优化器的优化对象是所有的参数w
optimizer = optim.Adadelta(net.parameters(), lr=0.001)
# 保存权重的路径
save_path = './models/AlexNet.pth'
# 只保存训练正确率最高的模型
best_acc = 0.0
for epoch in range(EPOCH):

    # train；dropout只适合用在训练集中，所以net.train()在训练集中开启dropout，net.eval在验证集中关闭
    net.train()
    # 训练过程中的平均损失
    running_loss = 0.0
    # t1 = time.perf_counter()
    for step, data in enumerate(train_loader, start=0):
        # 将数据分为图像和标签
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))  # 把训练图像指定到设备中

        #       计算预测值与真实值的损失，同时将labels指定到设备上
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        # print train process（打印训练进度）
        #      当前的训练步数 / 训练一轮所需要的步数 = rate （训练进度）
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()
    # 计算训练时间
    # print(time.perf_counter() - t1)
    # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    # 这是验证部分
    # with torch.no_grad():禁止pytorch对参数的跟踪，在验证过程中是不会计算损失梯度的
    with torch.no_grad():
        for data_test in validate_loader:
            test_images, test_labels = data_test  # 遍历验证机，划分为图片和标签
            outputs = net(test_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]

            #       将预测与真实标签进行对比，对的为1，错的为0进行求和--》求出正确的样本个数
            acc += (predict_y == test_labels.to(device)).sum().item()
        accurate_test = acc / val_num

        # 若当前准确率>历史最优准确率
        if accurate_test > best_acc:
            best_acc = accurate_test
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, acc / val_num))
        # 训练完成会达到最优参数
print('Finished Training')


