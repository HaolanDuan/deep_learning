import os
import random

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100

import pickle
import numpy as np

FOLDER_PATH = 'C:/Users/duanh/Desktop/eecs498/ai6103/assignment-mobilenet-code'

def unpickle(file):
    with open(file, 'rb') as fo:
        myDict = pickle.load(fo, encoding='latin1')
    return myDict

def _extract_tensors(dset, num=None, x_dtype=torch.float32):
    """
    Extract the data, coarse labels, and fine labels from a CIFAR100 dataset object
    and convert them to tensors.

    Input:
    - dset: A torchvision.datasets.CIFAR100 object
    - num: Optional. If provided, the number of samples to keep.
    - x_dtype: Optional. data type of the input image

    Returns:
    - X: 'x_dtype' tensor of shape (N, 3, 32, 32)
    - y: fine label int64 tensor of shape (N, )
    """
    x = torch.tensor(dset.data, dtype=x_dtype).permute(0,3,1,2).div_(255)
    y = torch.tensor(dset.targets, dtype=torch.int64)
    if num is not None:
        if num <= 0 or num > x.shape[0]:
            raise ValueError(
                "Invalid value num=%d; must be in the range [0, %d]"
                % (num, x.shape[0])
            )
        x = x[:num].clone()
        y = y[:num].clone()
    return x, y

def cifar100(num_train=None, num_test=None, x_dtype=torch.float32):
    """
    Return the CIFAR100 dataset, automatically downloading it if necessary.
    This function can also subsample the dataset.

    Inputs:
    - num_train: Optional. How many samples to keep from the training set.
    - num_test: Optional. How many samples to keep from the test set.

    Returns:
    - x_train: 'x_dtype' tensor of shape (num_train, 3, 32, 32)
    - y_train: int64 tensor of shape (num_train, 3, 32, 32)
    - x_test: 'x_dtype' tensor of shape (num_test, 3, 32, 32)
    - y_test: int64 tensor of shape (num_test, 3, 32, 32)
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])
    download = not os.path.isdir('cifar-100-python')
    dset_train = CIFAR100(root='.', download=download, train=True, transform=transform_train)
    dset_test = CIFAR100(root='.', train=False, transform=transform_test)
    x_train, y_train = _extract_tensors(dset_train, num_train, x_dtype)
    x_test, y_test = _extract_tensors(dset_test, num_test, x_dtype)
    return x_train, y_train, x_test, y_test

def compute_mean_std(train_data):
    """
    Input:
    - train_data: in the shape of (num_train, 3, 32,32)

    Returns:
    - mean: in the shape of (3, ). Numbers are for RGB
    - std: in the shape of (3, ). Numbers are for RGB
    """
    mean = torch.mean(train_data, dim=(0,2,3))
    std = torch.std(train_data, dim=(0,2,3))
    return mean, std

def preprocess_cifar100(cuda=True, bias_trick=False,
                    flatten=True, validation_ratio=0.2, dtype=torch.float32):
    X_train, y_train, X_test, y_test = cifar100(x_dtype=dtype)

    if cuda:
        X_train = X_train.cuda()
        y_train = y_train.cuda()
        X_test = X_test.cuda()
        y_test = y_test.cuda()

    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        
    num_training = int(X_train.shape[0] * (1.0 - validation_ratio))
    num_validation = X_train.shape[0] - num_training

    data_dict = {} 
    
    data_dict["X_train"] = X_train[0:num_training]
    data_dict["y_train"] = y_train[0:num_training]

    data_dict["X_val"] = X_train[num_training:num_training+num_validation]
    data_dict["y_val"] = y_train[num_training:num_training+num_validation]
    
    data_dict["X_test"] = X_test
    data_dict["y_test"] = y_test


    # For random split:
    # train_set = MyMNIST(root=self.root, train=True, transform=transform, download=False)
    # index_sub = np.random.choice(np.arange(len(train_set)), 10000, replace=False)
    # train_set.data = train_set.data[index_sub]
    # train_set.targets = train_set.targets[index_sub]
    # train_set.semi_targets = train_set.semi_targets[index_sub]

    return data_dict

