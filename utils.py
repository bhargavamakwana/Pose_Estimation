from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, random_split
from torch.utils.data.sampler import Sampler, BatchSampler
from scipy.spatial.transform import Rotation as R
import glob
import cv2


class PoseDataset(Dataset):
    """
    Custom class to load data
    """
    def __init__(self, X_train, labels=None, transform=None, batch_size=16):
        self.x_train = []
        self.transform = transform
        self.window = batch_size
        self.count = 0
        for img, label in zip(X_train, labels):
            self.x_train.append([img[0], img[1], label])

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, index):
        x_imgs = []
        x_depths = []
        x_labels = []
        for i in range(index, index+self.window):
            img = self.x_train[index][0]
            depth = self.x_train[index][1]
            label = torch.from_numpy(self.x_train[index][2])
            if self.transform:
                img = self.transform(img)
                depth = self.transform(depth)
            x_imgs.append(img)
            x_depths.append(depth)
            x_labels.append(label)

        x_imgs = torch.stack(x_imgs)
        x_depths = torch.stack(x_depths)
        x_labels = torch.stack(x_labels)
        x_imgs = torch.squeeze(x_imgs, 0)
        x_depths = torch.squeeze(x_depths, 0)
        x_labels = torch.squeeze(x_labels, 0)
        #print(f'Window Size:  {index} {index + self.window}')

        return {'image': x_imgs, 'depth': x_depths, 'label': x_labels}




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
        X_train.append([cv2.resize(cv2.imread(img), (224,224)), cv2.resize(cv2.imread(depth), (224,224))])
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
        The transform is calculated with the help of following logic:
        P = r1.T * r2     r1.T(t1 - t2)
            0             1

        P1 = r1     t1
             0      1


        P2 = r2     t2
             0      1

        """
        labels = []
        for h0, h1 in zip(poses, poses[1:]):
            r0, t0 = h0[:3, :3], h0[:3, 3]
            r1, t1 = h1[:3, :3], h1[:3, 3]
            r0_r1 = r0.dot(r1.T)
            t0_t1 = np.dot(r1.T, t0 - t1)
            r = R.from_matrix(r0_r1)
            final_r_t = np.append(r.as_quat(), t0_t1)
            labels.append(final_r_t)
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
    train_set = torch.utils.data.Subset(train_data, range(int(train_len * 0.8)))
    val_set = torch.utils.data.Subset(train_data, range(int(train_len * 0.8), train_len))
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=1,
                                              num_workers=2, drop_last=True)
    testloader = torch.utils.data.DataLoader(val_set, batch_size=1,
                                             num_workers=2, drop_last=True)
    return trainloader, testloader
