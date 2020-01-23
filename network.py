import torch
import torch.nn as nn
from function import *
from torch.autograd import Variable
import sys
import numpy as np


class Conv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_s):
        super(Conv2d, self).__init__()
        padding = int((kernel_s-1)/2)
        self.conv = nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=kernel_s,
            stride=1,
            padding=padding
        )
        self.relu = nn.ReLU(inplace=True)#节约一下

    def forward(self, inputfile):
        x = self.conv(inputfile)
        x = self.relu(x)
        return x

class mcnn(nn.Module):
    def __init__(self):
        super(mcnn, self).__init__()
        self.net_s = nn.Sequential(
            Conv2d(1, 16, 9),
            nn.MaxPool2d(2),
            Conv2d(16, 32, 7),
            nn.MaxPool2d(2),
            Conv2d(32, 16, 7),
            Conv2d(16, 8, 7)
        )
        self.net_m = nn.Sequential(
            Conv2d(1, 20, 7),
            nn.MaxPool2d(2),
            Conv2d(20, 40, 5),
            nn.MaxPool2d(2),
            Conv2d(40, 20, 5),
            Conv2d(20, 10, 5)
        )
        self.net_l = nn.Sequential(
            Conv2d(1, 24, 5),
            nn.MaxPool2d(2),
            Conv2d(24, 48, 3),
            nn.MaxPool2d(2),
            Conv2d(48, 24, 3),
            Conv2d(24, 12, 3)
        )
        self.creat_dp = nn.Sequential(
            Conv2d(30, 1, 1)
        )

    def forward(self, x):
        ns = self.net_s(x)
        nm = self.net_m(x)
        nl = self.net_l(x)
        tempNet = torch.cat((ns, nm, nl), 1)
        tempNet = self.creat_dp(tempNet)
        return tempNet


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.netw = mcnn()
        self.loss_func = nn.MSELoss()
    @property
    def loss(self):
        return self.lossf
    def forward(self, img_data, csv_data=None):
        data = np2value(img_data, is_train=self.training)
        dMap = self.netw(data)
        if self.training:
            csv_data = np2value(csv_data, is_train=self.training)
            self.lossf = self.loss_func(dMap, csv_data)

        return dMap
def evaluate(model,val_data):
    mae = 0.0
    mse = 0.0
    net = Net()
    net.load_state_dict(torch.load(model))
    net.cuda()
    net.eval()
    for i in val_data:
        data = i['data']
        trueData = i['csvData']
        dMap = net(data,trueData)
        dMap = dMap.data.cpu().numpy()
        trueCount = np.sum(trueData)
        predictCount = np.sum(dMap)
        mae += abs(trueCount-predictCount)
        mse += (trueCount-predictCount)**2
    mae = mae/val_data.getNum()
    mse = np.sqrt(mse/val_data.getNum())
    return mae ,mse