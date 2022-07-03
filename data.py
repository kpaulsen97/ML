from numpy import load
import torch
import numpy as np
from torchvision import datasets, transforms


def mnist():
    train = []
    labels_train = []
    test = []
    labels_test = []
    for i in range(4):
        
        string = "train_{i}.npz"
        data = load(string.format(i=i))
        labels = np.array(data["labels"])
        data = np.array(data["images"])
        train.append(data.reshape(5000,784))
        labels_train.append(labels)
        
    data = load("test.npz")
    test = np.array(data["images"]).reshape(5000,784)
    labels_test = torch.from_numpy(np.array(data["labels"]).reshape(5000,))
    # exchange with the corrupted mnist dataset
    train = torch.from_numpy(np.array(train).reshape(20000,784))
    labels_train = torch.from_numpy(np.array(labels_train).reshape(20000,))
    test = torch.from_numpy(np.array(test)) 
    # Define a transform to normalize the data
    
    # Download and load the training data
    #trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
    #trainloader = torch.utils.data.DataLoader([train, labels_train], batch_size=64, shuffle=True)

    # Download and load the test data
    #testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
    #testloader = torch.utils.data.DataLoader([test, labels_test], batch_size=64, shuffle=True)
    
    return [train,labels_train], [test, labels_test]


a,_=mnist()


