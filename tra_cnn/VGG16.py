import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import pandas as pd
import csv
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):

    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.conv = nn.Sequential(
            nn.AvgPool2d(8, 7)
        )
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 2)

    def forward(self, x):
        out = self.conv(x)
        out = self.features(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


BARCH_SIZE = 32
LR = 0.001
EPOCH = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize(size=(227, 227)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

train_dataset = torchvision.datasets.ImageFolder(root='./data_pictures_4880/train/', transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BARCH_SIZE, shuffle=True)

validation_dataset = torchvision.datasets.ImageFolder(root='./data_pictures_4880/valid/',
                                                      transform=transform)

validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=BARCH_SIZE, shuffle=True)

vggNet = VGG('VGG11').to(device)
criterion = nn.CrossEntropyLoss().to(device)
opti = torch.optim.Adadelta(vggNet.parameters(), lr=LR)


if __name__ == '__main__':
    path='train_final'
    Accuracy_list = []
    Loss_list = []
    F1=[]
    Recall=[]
    Precision=[]

    #训练
    los_list=[]
    acc_list=[]
    for epoch in range(EPOCH):
        sum_loss = 0.0
        correct1 = 0

        total1 = 0
        FN=0
        FP=0
        TP=0
        TN=0
        for i, (images, labels) in enumerate(train_loader):
            # print(i)
            num_images = images.size(0)
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            out = vggNet(images)

            loss = criterion(out, labels)
            sum_loss += loss.item()

            opti.zero_grad()
            loss.backward()
            opti.step()

            if i % 10 == 9:
                print('[%d, %d] loss: %.03f'
                      % (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0

            _, predicted = torch.max(out.data, 1)

            total1 += labels.size(0)
            # print('predict={},labels={}'.format(predicted.size(), labels.size()))
            correct1 += (predicted == labels).sum().item()
            train_correct01 = ((predicted.data == 0) & (labels.data == 1)).sum().item()  # 原标签为1，预测为0
            train_correct10 = ((predicted.data == 1) & (labels.data == 0)).sum().item()
            train_correct11 = ((predicted.data == 1) & (labels.data == 1)).sum().item()
            train_correct00 = ((predicted.data == 0) & (labels.data == 0)).sum().item()

            FP += train_correct01
            FN += train_correct10
            TN += train_correct11
            TP += train_correct00

        if(TP==0):
            TP=1
            P = TP / (TP + FP)  # 精确度
            R = TP / (TP + FN)  # 召回率
            f1 = 2 * R * P / (R + P)  # F值：P和R的调和平均值
            A = (TP + TN) / (TP + TN + FP + FN)  # 准确率
        else:
            P = TP / (TP + FP)  # 精确度
            R = TP / (TP + FN)  # 召回率
            f1 = 2 * R * P / (R + P)  # F值：P和R的调和平均值
            A = (TP + TN) / (TP + TN + FP + FN)  # 准确率

        Accuracy_list.append(100.0 * correct1 / total1)
        Precision.append(100*P)
        Recall.append(100*R)
        F1.append(100*f1)

        print('accurary={}'.format(100.0 * correct1 / total1))
        print(A)
        print(P)
        print(R)
        print(f1)
        Loss_list.append(loss.item())

    torch.save(vggNet, './alexNet.pkl')



    # result={
    #     'name':[],
    #     'loss':[],
    #     'accuracy':[],
    #     'f1':[],
    #     'recall':[],
    #     'precision':[]
    #
    # }
    #
    # result['name'].append(str(EPOCH)+str(path))
    # result['accuracy'].append(Accuracy_list)
    # result['loss'].append(Loss_list)
    # result['f1'].append(F1)
    # result['recall'].append(Recall)
    # result['precision'].append(Precision)
    # print(result)

    # f = open('result.csv', 'a', newline='',encoding='utf-8')
    # csv_writer = csv.writer(f)
    # s=str(EPOCH)+str(path)
    # result=[[s],[Loss_list],[Accuracy_list],[F1],[Recall],[Precision]]
    # csv_writer.writerow(result)
    # f.close()


    # reslt=pd.DataFrame(result)
    # reslt.to_csv('result.csv',index=None)

    # x1 = range(0, EPOCH)
    # x2 = range(0, EPOCH)
    # y1 = Accuracy_list
    # y2 = Loss_list
    # plt.subplot(2, 1, 1)
    # plt.plot(x1, y1, 'o-')
    # plt.title('Train accuracy vs. epoches')
    # plt.ylabel('Train accuracy')
    # plt.subplot(2, 1, 2)
    # plt.plot(x2, y2, '.-')
    # plt.xlabel('Train loss vs. epoches')
    # plt.ylabel('Train loss')
    # # plt.savefig("accuracy_epoch" + (str)(EPOCH) +path+ ".png")
    # plt.show()
    #
    # x3 = range(0, EPOCH)
    # x4 = range(0, EPOCH)
    # x5=range(0,EPOCH)
    # y3=Precision
    # y4=Recall
    # y5=F1
    # plt.subplot(3,1,1)
    # plt.plot(x3, y3, 'o-')
    # plt.title('Train Precision vs. epoches')
    # plt.ylabel('Train Precision')
    # plt.subplot(3, 1, 2)
    # plt.plot(x4, y4, '.-')
    # plt.xlabel('Train Recall vs. epoches')
    # plt.ylabel('Train recall')
    # plt.subplot(3, 1, 3)
    # plt.plot(x4, y4, '.-')
    # plt.xlabel('Train F1 vs. epoches')
    # plt.ylabel('Train f1')
    # # plt.savefig("f1_epoch" + (str)(EPOCH) +path+ ".png")
    # plt.show()



