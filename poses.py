import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
from utils import train_load_util
import matplotlib.pyplot as plt
import glob
import cv2
import pickle

# Hyper-parametrs
batch_size = 1
epochs = 2


class Base(nn.Module):
    """
    Base Model class
    """
    def __init__(self, inp_size, op_size, stride=2):
        super(Base, self).__init__()
        self.conv1 = nn.Conv2d(inp_size, op_size, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(op_size)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(op_size, op_size//2, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(op_size//2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.linear1 = nn.Linear(5408, 512)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = F.relu(x)
        x = self.pool1(x)
        x = self.bn2(self.conv2(x))
        x = F.relu(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        return x.float()



class NetGated(nn.Module):
    """
    NetGated class which has Base class initialized with RGB and Depth Images
    """

    def __init__(self, block, inp_size, dim, batch_szie = 16):
        super(NetGated, self).__init__()

        self.layer1 = block(inp_size, dim)
        self.layer2 = block(inp_size, dim)
        self.layer3 = nn.Linear(1024, 64)
        self.layer4 = nn.Linear(64, 2)
        self.layer5 = nn.Linear(512, 7)

    def forward(self, rgb, depth):
        x_rgb = self.layer1(rgb)
        x_depth = self.layer2(depth)
        x = torch.cat([x_rgb, x_depth], dim=1)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.float()
        embed_x = torch.mul(torch.reshape(x[:, 0], (x.shape[0], 1)), x_rgb) + torch.mul(torch.reshape(x[:, 1], (x.shape[0], 1)), x_depth)
        out = self.layer5(embed_x)
        return out.float(), embed_x



class Attention(nn.Module):
    """
    Temporal Attention
    """
    def __init__(self, NetGated, block, inp_size, dim, batch_size=batch_size):
        super(Attention, self).__init__()
        self.net_gated = NetGated(block, inp_size, dim, batch_size)
        self.linear1 = nn.Linear(8192, 512)
        self.linear2 = nn.Linear(1024, 16)
        self.linear3 = nn.Linear(1024, 7)
        self.embed = None
    def forward(self, rgb, depth):
        _, self.embed = self.net_gated(rgb, depth)
        curr_embed = self.embed[-1]
        curr_embed = torch.unsqueeze(curr_embed, dim=0)
        flatten_embed = torch.flatten(self.embed)
        x = self.linear1(flatten_embed)
        x = torch.unsqueeze(x, dim=0)
        x = torch.cat([x, curr_embed], dim=1)
        x = self.linear2(x)
        x = torch.matmul(x, self.embed)
        x = torch.cat([x, curr_embed], dim=1)
        x = self.linear3(x)
        return x.float()


# Initialize Optimizer and Loss function objects


net = Attention(NetGated, Base, 3, 64, batch_size=16)
loss_fn = torch.nn.MSELoss()
optim = torch.optim.Adam(net.parameters(), lr=1e-8)
mae_loss =  torch.nn.L1Loss()
def r2_loss(pred, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target-target_mean)**2)
    ss_res = torch.sum((target-pred)**2)
    r2 = 1- ss_res/ss_tot
    return r2


train_loss_list = []
test_loss_list = []

test_mae_list = []
test_r2_list = []
pred_list = []
label_list = []
for i in range(1, 2):
    base_path = "./chess/seq-0" + str(i) + "/"
    trainloader, testloader = train_load_util(base_path, batch_size)
    for epoch in range(epochs):
        print(f'Training for Epoch {epoch}')
        for j, data in enumerate(trainloader):
            timgs, tdepths, labels = data['image'], data['depth'], data['label']
            timgs = torch.squeeze(timgs, 0)
            tdepths = torch.squeeze(tdepths, 0)
            labels = torch.squeeze(labels, 0)
            pred = net(timgs, tdepths)
            loss = loss_fn(pred, labels.float()[-1])
            loss.backward()
            optim.step()
            if j % 2 == 0:
                loss = loss.item()
                print(f'Train loss: {loss:>7f}')
                train_loss_list.append(loss)

    best_loss = float("inf")
    print(f'Testing Started for Epoch {epoch}')
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            timgs, tdepths, labels = data['image'], data['depth'], data['label']
            timgs = torch.squeeze(timgs, 0)
            tdepths = torch.squeeze(tdepths, 0)
            labels = torch.squeeze(labels, 0)
            pred = net(timgs, tdepths)
            pred_list.append(pred.detach().numpy())
            label_list.append(labels.float().detach().numpy())

            loss = loss_fn(pred, labels.float()[-1])
            if i % 2 == 0:
                loss = loss.item()
                best_loss = min(best_loss, loss)
                test_loss_list.append(loss)
        print(f'##################### Test loss: {best_loss:>7f} Acc: {100*abs(best_loss/sum(labels.float().ravel()))} ######################')
    with open("pred_list_" + str(i) + ".pkl", "wb+") as f:
        pickle.dump(pred_list, f)

    with open("label_list_" + str(i) + ".pkl", "wb+") as f:
        pickle.dump(label_list, f)

#plt.plot(loss_list, color='magenta', marker='o',mfc='pink' ) #plot the data
#plt.ylabel('Train Loss') #set the label for y axis
#plt.xlabel('indexes') #set the label for x-axis
#plt.show() #display the graph
