#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 07:45:44 2019

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


train_data = datasets.MNIST(root = '/Applications/Queens MMA/My Code/Pytorch/MNIST/Data', train =True, download = True, transform = transform)
test_data = datasets.MNIST(root = '/Applications/Queens MMA/My Code/Pytorch/MNIST/Data', train =False, download = True, transform = transform)

train_loader = DataLoader(train_data, batch_size = 10, shuffle = True)
train_loader = DataLoader(train_data, batch_size = 10, shuffle = False)


class ConvolutionalNetwork(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels =6, kernel_size=3, stride = 1) #1 In channel, 
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels =16, kernel_size=3, stride = 1) #1 In channel, 
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        
    def forward(self,X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X,2,2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X,2,2)
        X = X.view(-1, 16*5*5)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        
        return F.log_softmax(X, dim =1)
    
torch.manual_seed(42)

model = ConvolutionalNetwork()    
model        


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)    

import time

start_time = time.time()

epochs = 5

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
        
        
plt.plot(train_loss, label = 'train loss')
plt.plot(test_loss, label = 'validation loss')  
plt.title('Loss At Epoch')
plt.legend()
plt.show()  

        
        
test_load_all  = DataLoader(test_data, batch_size= 10000, shuffle = False)

with torch.no_grad():
    correct =0

    for X_test, y_test in test_load_all:
        y_val = model(X_test)
        predicted = torch.max(y_val.data,1)[1]
        correct = (predicted == y_test).sum()
        
        
100*(correct.item() /10000)


with torch.no_grad():
    new_pred = model(test_data[333][0].view(1,1,28,28))        
    
new_pred.argmax().item()
        
        
        
        
        
        
        
        
        
        
        
        