import torch
# from alexnet_model import AlexNet
import torch.nn as nn
from alexnet import AlexNet
from torchvision import transforms, datasets, utils
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.models as models
import torchvision.transforms as transforms
from torchnet import meter

#文件路径
image_path='../save/cnn_save/Grid200/abnormal_5percent_pic'

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


test_dataset = datasets.ImageFolder(root=image_path + "/test",
                                        transform=data_transform)
test_num = len(test_dataset)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=16, shuffle=True,
                                              num_workers=0)


model=AlexNet(num_classes=2)
model_weight_path = "./models/AlexNet.pth"
model.load_state_dict(torch.load(model_weight_path))
model.eval()


correct = 0
total = 0
acc = 0.0

confusion_matrix = meter.ConfusionMeter(2)
for i, sample in enumerate(test_loader):
    inputs, labels = sample[0], sample[1]
    outputs = model(inputs)

    _, prediction = torch.max(outputs, 1)
    correct += (labels == prediction).sum().item()
    total += labels.size(0)
    confusion_matrix.add(outputs.detach().squeeze(),labels.long())

acc = correct / total
print('test finish, total:{}, correct:{}, acc:{:.3f}'.format(total, correct, acc))
cm_value=confusion_matrix.value()
print(cm_value)

TP=cm_value[0][0]
FP=cm_value[0][1]
FN=cm_value[1][0]
TN=cm_value[1][1]
acc=(TP+TN)/(TP+TN+FP+FN)
pre=TP/(TP+FP)
recall=TP/(TP+FN)
F1=(2*pre*recall)/(pre+recall)
# 保存实验结果
result_save_path=image_path+'\\'+'alexnet_result.txt'
with open(result_save_path, 'w', encoding='utf-8') as f:
    f.write('acc={} | precision={} | recall={} | f1={}\n'.
            format(acc, pre, recall, F1))
