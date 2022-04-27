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

# Hyper-parametrs
batch_size = 16
epochs = 20


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
        self.layer5 = nn.Linear(512, 12)

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
        self.linear3 = nn.Linear(1024, 12)
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


class PoseDataset(Dataset):
    """
    Custom class to load data
    """
    def __init__(self, X_train, labels=None, transform=None):
        self.x_train = []
        self.transform = transform
        for img, label in zip(X_train, labels):
            self.x_train.append([img[0], img[1], label])
    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, index):
        img = self.x_train[index][0]
        depth = self.x_train[index][1]
        if self.transform:
            x = self.transform(img)
            x_depth = self.transform(depth)
        return {'image': x, 'depth': x_depth, 'label': torch.from_numpy(self.x_train[index][2])}



files_color = glob.glob("./chess/seq-01/*.color.png")
files_depth = glob.glob("./chess/seq-01/*.depth.png")
files_labels = glob.glob("./chess/seq-01/*.txt")

files_color.sort()
files_depth.sort()
files_labels.sort()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# read and convert images to numpy array as well as labels.
def read_rgb_and_depth_to_numpy(rgb_files, depth_files):
    X_train = []
    for img, depth in zip(rgb_files, depth_files):
        X_train.append([cv2.resize(cv2.imread(img), (224,224)), cv2.resize(cv2.imread(img), (224,224))])
    return np.array(X_train)

def generate_labels(labels_list):
    """
    Utility Function to generate the labels.
    Find the relative pose and append matrix elements as labels.
    """

    labels = []
    poses = []
    def relative_rotations(poses):
        """
        find relative rotation between two frames
        """
        labels = []
        for r0, r1 in zip(poses, poses[1:]):
            r0_r1 = r1.dot(r0.T)
            r0_r1 = r0_r1.ravel()[:12]
            labels.append(r0_r1)
        return labels
    for i in range(len(files_labels)):
        label_list = []
        pose = np.loadtxt(files_labels[i])
        poses.append(pose)

    labels = relative_rotations(poses)

    return np.array(labels)





# Initialize Optimizer and Loss function objects


#net = NetGated(Base, 3, 64)
net = Attention(NetGated, Base, 3, 64, batch_size=16)
loss_fn = torch.nn.MSELoss()
optim = torch.optim.Adam(net.parameters(), lr=1e-8)

train_loss_list = []
test_loss_list = []

for i in range(1, 7):
    base_path = "./chess/seq-0" + str(i) + "/"
    trainloader, testloader = train_load_util(base_path, batch_size)
    for epoch in range(epochs):
        print(f'Training for Epoch {epoch}')
        for i, data in enumerate(trainloader, 0):
            timgs, tdepths, labels = data['image'], data['depth'], data['label']
            pred = net(timgs, tdepths)
            loss = loss_fn(pred, labels.float())
            loss.backward()
            optim.step()

            if i % 2 == 0:
                loss = loss.item()
                print(f'Train loss: {loss:>7f}')
                train_loss_list.append(loss)

        best_loss = float("inf")
        with torch.no_grad():
            for i, data in enumerate(trainloader, 0):
                timgs, tdepths, labels = data['image'], data['depth'], data['label']
                pred = net(timgs, tdepths)
                loss = loss_fn(pred, labels.float())
                if i % 2 == 0:
                    loss = loss.item()
                    best_loss = min(best_loss, loss)
                    test_loss_list.append(loss)
        print(f'##################### Test loss: {best_loss:>7f} ######################')
plt.plot(loss_list, color='magenta', marker='o',mfc='pink' ) #plot the data
plt.ylabel('Train Loss') #set the label for y axis
plt.xlabel('indexes') #set the label for x-axis
plt.show() #display the graph
