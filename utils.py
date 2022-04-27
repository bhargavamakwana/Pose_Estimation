from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, random_split
import glob
import cv2






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




def get_files_list(base_path):
    """
    """
    files_color = glob.glob(base_path + "/*.color.png")
    files_depth = glob.glob(base_path + "/*.depth.png")
    files_labels = glob.glob(base_path + "/*.txt")

    files_color.sort()
    files_depth.sort()
    files_labels.sort()

    return files_color, files_depth, files_labels



transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# read and convert images to numpy array as well as labels.
def read_rgb_and_depth_to_numpy(rgb_files, depth_files):
    X_train = []
    for img, depth in zip(rgb_files, depth_files):
        X_train.append([cv2.resize(cv2.imread(img), (224,224)), cv2.resize(cv2.imread(img), (224,224))])
    return np.array(X_train)

def generate_labels(files_labels):
    """
    Utility Function to generate the labels.
    Find the relative pose and append matrix elements as labels.
    """
    labels = []
    poses = []
    def relative_rotations(poses):
        """
        Find relative rotation between two frames
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


def train_load_util(base_path, batch_size):
    """
    Load Color and Depth Images along with the labels
    """
    files_color, files_depth, files_labels = get_files_list(base_path)
    X_train = read_rgb_and_depth_to_numpy(files_color, files_depth)
    train_labels = generate_labels(files_labels)

    # Load dataset into Numpy array.
    train_data = PoseDataset(X_train, train_labels, transform=transform)
    train_len = len(train_data)
    train_set, val_set = random_split(train_data, [int(train_len * 0.8), train_len - int(train_len * 0.8)])
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=True, num_workers=2, drop_last=True)
    testloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                              shuffle=True, num_workers=2, drop_last=True)
    return trainloader, testloader
