#!/usr/bin/env python3
# coding=utf-8

## Zpracování záznamu cílové kamery
# @file moduls.py
# @brief Architektury konvolučních neuronových sítí, které jsem zkoušel
# @author Jahoda Vojtěch

__author__ = "Jahoda Vojtěch"
__description__ = "Architektury konvolučních neuronových sítí, které jsem zkoušel"


import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as TModels


class CNN(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 1):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv4 = nn.Conv2d(in_channels = 32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))


        self.fc1 = nn.Linear(6400, 2048)
        self.fc2 = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        #print(x.reshape(x.shape[0], -1).shape)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
  
class CNN_dropoutV1(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 1):
        super(CNN_dropoutV1,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv4 = nn.Conv2d(in_channels = 32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))


        self.fc1 = nn.Linear(6400, 2048)
        self.fc2 = nn.Linear(2048, num_classes)

        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        #print(x.reshape(x.shape[0], -1).shape)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.dropout(self.fc1(x)))
        x = self.fc2(x)

        return x

class CNN_dropoutV2(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 1):
        super(CNN_dropoutV2,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv4 = nn.Conv2d(in_channels = 32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))


        self.fc1 = nn.Linear(6400, 2048)
        self.fc2 = nn.Linear(2048, num_classes)

        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        #print(x.reshape(x.shape[0], -1).shape)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
  
class CNN_dropoutV3(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 1):
        super(CNN_dropoutV3,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv4 = nn.Conv2d(in_channels = 32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))


        self.fc1 = nn.Linear(6400, 2048)
        self.fc2 = nn.Linear(2048, num_classes)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.25)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        #print(x.reshape(x.shape[0], -1).shape)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return x

class CNN_dropoutV4(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 1):
        super(CNN_dropoutV4,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv4 = nn.Conv2d(in_channels = 32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))


        self.fc1 = nn.Linear(6400, 2048)
        self.fc2 = nn.Linear(2048, num_classes)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout2(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout2(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout2(x)

        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout2(x)
        
        #print(x.reshape(x.shape[0], -1).shape)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)

        return x

class CNN_dropoutV5(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 1):
        super(CNN_dropoutV5,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv4 = nn.Conv2d(in_channels = 32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))


        self.fc1 = nn.Linear(6400, 2048)
        self.fc2 = nn.Linear(2048, num_classes)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout2(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout2(x)
        
        #print(x.reshape(x.shape[0], -1).shape)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)

        return x

class CNN_dropoutV6(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 1):
        super(CNN_dropoutV6,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv4 = nn.Conv2d(in_channels = 32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))


        self.fc1 = nn.Linear(6400, 2048)
        self.fc2 = nn.Linear(2048, num_classes)

        self.dropout1 = nn.Dropout(0.7)
        self.dropout2 = nn.Dropout(0.35)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = F.relu(self.conv4(x))
        x = self.pool(x)
        
        #print(x.reshape(x.shape[0], -1).shape)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return x

class CNN_strip(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 1):
        super(CNN_strip,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool1 = nn.MaxPool2d((2,2), stride=(2,2))

        self.conv2 = nn.Conv2d(in_channels = 8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool2 = nn.AvgPool2d((1,4), stride=(1,4))

        self.conv3 = nn.Conv2d(in_channels = 16, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool3 = nn.MaxPool2d((2,2), stride=(2,2)) 

        self.conv4 = nn.Conv2d(in_channels = 32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool4 = nn.AvgPool2d((1,2), stride=(1,2))

        self.conv5 = nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool5 = nn.AvgPool2d((1,2), stride=(1,2))


        self.fc1 = nn.Linear(5120, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x) #[80,80]

        x = F.relu(self.conv2(x))
        x = self.pool2(x) #[40,16]

        x = F.relu(self.conv3(x))
        x = self.pool3(x) #[40,8]

        x = F.relu(self.conv4(x))
        x = self.pool4(x) #[40,4]

        x = F.relu(self.conv5(x))
        x = self.pool5(x) #[40,2]

        x = x.reshape(x.shape[0], -1) #[64,40,2]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class CNN_strip_dropout(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 1):
        super(CNN_strip_dropout,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool1 = nn.MaxPool2d((2,2), stride=(2,2))

        self.conv2 = nn.Conv2d(in_channels = 8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool2 = nn.AvgPool2d((1,4), stride=(1,4))

        self.conv3 = nn.Conv2d(in_channels = 16, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool3 = nn.MaxPool2d((2,2), stride=(2,2)) 

        self.conv4 = nn.Conv2d(in_channels = 32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool4 = nn.AvgPool2d((1,2), stride=(1,2))

        self.conv5 = nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool5 = nn.AvgPool2d((1,2), stride=(1,2))


        self.fc1 = nn.Linear(5120, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.25)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x) #[80,80]

        x = F.relu(self.conv2(x))
        x = self.pool2(x) #[40,16]

        x = F.relu(self.conv3(x))
        x = self.pool3(x) #[40,8]

        x = F.relu(self.conv4(x))
        x = self.pool4(x) #[40,4]

        x = F.relu(self.conv5(x))
        x = self.pool5(x) #[40,2]

        x = self.dropout1(x)
        x = x.reshape(x.shape[0], -1) #[64,40,2]
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return x

class VGGNetSeq(nn.Module):
  def __init__(self, base_channels=3, num_classes=1):
    super(VGGNetSeq, self).__init__()
    '''self.conv_layers = nn.Sequential(
      nn.Conv2d(3, base_channels*1, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),

      nn.Conv2d(base_channels*1, base_channels*2, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),

      nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )'''
    model = TModels.vgg13(pretrained=True)
    self.conv_layers = model.features[:10]
    
    self.conv5 = nn.Conv2d(in_channels = 128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
    self.conv6 = nn.Conv2d(in_channels = 128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
    self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

    self.fc1 = nn.Linear(12800, 4096)
    self.fc2 = nn.Linear(4096, num_classes)
   
  def forward(self, x):
    out = self.conv_layers(x)  #[16, 32, 4, 4]
    out = F.relu(self.conv5(out))
    out = self.pool(out)
    out = F.relu(self.conv6(out))
    out = self.pool(out)
    out = out.reshape(out.size(0), -1)  # [16, 4*4*32] [16, 512]
    out = F.relu(self.fc1(out))
    out = self.fc2(out)
    return out

class VGGNetSeq_dropout(nn.Module):
  def __init__(self, base_channels=3, num_classes=1):
    super(VGGNetSeq_dropout, self).__init__()
    '''self.conv_layers = nn.Sequential(
      nn.Conv2d(3, base_channels*1, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),

      nn.Conv2d(base_channels*1, base_channels*2, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),

      nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )'''
    model = TModels.vgg13(pretrained=True)
    self.conv_layers = model.features[:10]
    
    self.conv5 = nn.Conv2d(in_channels = 128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
    self.conv6 = nn.Conv2d(in_channels = 128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
    self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

    self.fc1 = nn.Linear(12800, 4096)
    self.fc2 = nn.Linear(4096, num_classes)
   
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.25)

  def forward(self, x):
    out = self.conv_layers(x)  #[16, 32, 4, 4]
    out = F.relu(self.conv5(out))
    out = self.pool(out)
    out = F.relu(self.conv6(out))
    out = self.pool(out)
    out = out.reshape(out.size(0), -1)  # [16, 4*4*32] [16, 512]
    out = self.dropout1(out)
    out = F.relu(self.fc1(out))
    out = self.dropout2(out)
    out = self.fc2(out)
    return out