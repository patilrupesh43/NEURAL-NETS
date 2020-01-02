#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 09:03:12 2019

@author: rupesh
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


transform = transforms.ToTensor()


train_data = datasets.CIFAR10(root = '/Applications/Queens MMA/My Code/Pytorch/CIFAR/Data', train =True, download = True, transform = transform)
test_data = datasets.CIFAR10(root = '/Applications/Queens MMA/My Code/Pytorch/CIFAR/Data', train =False, download = True, transform = transform)

train_data
test_data

torch.manual_seed(101)

train_loader = DataLoader(train_data, batch_size =10, shuffle = True)
test_loader = DataLoader(test_data, batch_size =10, shuffle = False)




class ConvolutionalNetwork(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 6, 5,1)
        self.conv2 = nn.Conv2d(6,16,3,1)
        self.fc1 = nn.Linear(6*6*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        
    def forward(self,X):
        
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X,2,2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X,2,2)
        X = X.view(-1, 6*6*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        
        return F.log_softmax(X, dim = 1)
    

torch.manual_seed(101)
model = ConvolutionalNetwork()        
model

criterion = nn.CrossEntropyLoss()        
optimizer = torch.optim.Adam(model.parameters(), lr =0.001)        

import time

start_time = time.time()

epochs = 10

train_loss = []
test_loss = []
train_correct = []
test_correct = []

for i in range(epochs):
    
    trn_corr = 0
    test_corr = 0
    
    for b, (X_train, y_train) in enumerate(train_loader):
        b+= 1
        
        y_pred = model(X_train)
        loss = criterion(y_pred,y_train)
        
        predicted = torch.max(y_pred.data,1)[1]
        batch_correct = (predicted == y_train).sum()
        trn_corr += batch_correct
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if b%600 == 0:
            print(f'EPOCH: {i} BATCH {b} LOSS: {loss.item()}')
            
    train_loss.append(loss)
    train_correct.append(trn_corr)
    
    
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(train_loader):
            b+= 1
            
            y_val = model(X_test)
           # 
            
            predicted = torch.max(y_val.data,1)[1]
            batch_correct = (predicted == y_test).sum()
            
    loss = criterion(y_val,y_test)
    test_loss.append(loss)
    test_correct.append(test_corr)


total_time = time.time() - start_time
print(f'Training took {total_time/60} minutes')        
        
        
        
torch.save(model.state_dict(), 'Cfar_CNN.pt')
        
        
        
plt.plot(train_loss, label = 'train loss')
plt.plot(test_loss, label = 'validation loss')  
plt.title('Loss At Epoch')
plt.legend()
plt.show()  

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        