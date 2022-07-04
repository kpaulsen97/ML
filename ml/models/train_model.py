# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 15:08:26 2022

@author: Lenovo
"""

import torch
from torch import nn, optim
from data import mnist
from model import MyAwesomeModel
import matplotlib.pyplot as plt
from pathlib import Path
import os    



def train_model():
        
        os.environ['KMP_DUPLICATE_LIB_OK']='True'
        # TODO: Implement training loop here
        path = Path('C:\\Users\\Lenovo\\Desktop\\University\\Machine Learning Operations\\final_exercise\\ml\\data\\processed')
        model = MyAwesomeModel()
        images = torch.load(path/'train.pt')
        labels = torch.load(path/'labels_train.pt')
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.003)
        epochs = 8

        train_loss = []
        for e in range(epochs):
            #running_loss = 0
            
        
            optimizer.zero_grad()
        
            log_ps = model(images)
            
            loss = criterion(log_ps,labels)
            loss.backward()
            optimizer.step()
            
        
        
            train_loss.append(loss.item())
            
        path = Path('C:\\Users\\Lenovo\\Desktop\\University\\Machine Learning Operations\\final_exercise\\ml\\models')

        torch.save(model, path/'trained_model.pt')
            
        plt.plot(range(epochs),train_loss)
        path = Path('C:\\Users\\Lenovo\\Desktop\\University\\Machine Learning Operations\\final_exercise\\ml\\reports\\figures')
        plt.savefig(path/'training_curve.png')
        
train_model()