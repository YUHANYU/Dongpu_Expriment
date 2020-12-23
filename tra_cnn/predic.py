import torch
import os
from PIL import Image
from torchvision import transforms
from VGG16 import *

device = torch.device('cuda')
transform=transforms.Compose([
            transforms.Resize((227,227)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
                            ])
def prediect(img_path):
    net=torch.load('./alexNet.pkl')
    net=net.to(device)
    torch.no_grad()
    img=Image.open(img_path)
    img=transform(img).unsqueeze(0)
    img_ = img.to(device)
    outputs = net(img_)
    _, predicted = torch.max(outputs, 1)
    # print(predicted)
    print('this picture maybe :',predicted[0])

if __name__ == '__main__':
    path='./data_pictures_4880/test/normal/'
    folders = os.listdir(path)
    print(len(folders))
    for file in folders:
        source=path+str(file)
        # prediect('train_final/test/normal/1372639875620000009_0.jpg')
        prediect(source)
