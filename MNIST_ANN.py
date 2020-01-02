#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 20:21:35 2019

@author: rupesh
"""
#pip install torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import pandas as panda
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

transform = transforms.ToTensor()

train_data = datasets.MNIST(root = '/Applications/Queens MMA/My Code/Pytorch/MNIST/Data', train =True, download = True, transform = transform)

test_data = datasets.MNIST(root = '/Applications/Queens MMA/My Code/Pytorch/MNIST/Data', train =False, download = True, transform = transform)

type(train_data)

train_data[0]

image, label = train_data[0]
image.shape
label
    
plt.imshow(image.reshape((28,28)), cmap='gist_yarg')

torch.manual_seed(101)

train_loader = DataLoader(train_data, batch_size=100, shuffle = True) #shuffle to avoif all numbers together

test_loader = DataLoader(test_data, batch_size = 500, shuffle = False)

from torchvision.utils import make_grid

#FIRST BATCH
for images, label in train_loader:
    break

image.shape
label.shape

print('LAbels: ', label[:12].numpy())

im = make_grid(images[:12], nrow=15)
plt.figure(figsize =(10,8))
plt.imshow(np.transpose(im.numpy(), (1,2,0)))


class MultilayerPerceptron(nn.Module):
    
    def __init__(self, in_sz = 784, out_sz = 10, layers = [120,84]):
        
        super().__init__()
        
        self.fc1 = nn.Linear(in_sz,layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1],out_sz)
        
    
    def forward(self, X):
        
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        
        return F.log_softmax(X, dim =1)
    
    
torch.manual_seed(101)
model = MultilayerPerceptron()
        
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


import time

start_time = time.time()

epochs = 10

train_loss = []
test_loss = []

train_correct = []
test_correct = []

for i in range(epochs):
    
    trn_corr = 0
    tst_corr = 0
    
    for b, (X_train, y_train) in enumerate(train_loader):
        b+=1
        y_pred = model(X_train.view(100,-1))
        loss = criterion(y_pred,y_train)
        
        predicted = torch.max(y_pred.data,1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if b%200 ==0:
            accurracy = trn_corr.item() * 100 / (100*b)
            print(f'Epich {i} batch {b} loss:{loss.item()} accuracy')
            
    train_loss.append(loss)
    train_correct.append(trn_corr)
    
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            y_val = model(X_test.view(500,-1))
            
            predicted = torch.max(y_val.data,1)[1]
            trn_corr = (predicted == y_test).sum()
             
    
    loss = criterion(y_val, y_test)
    test_loss.append(loss)
    test_correct.append(tst_corr)
    




total_time = start_time - time.time()
print(f'Duration  {total_time/60} mins')



plt.plot(train_loss, label='Training Loss')
plt.plot(test_loss, label='Test Loss')
plt.legend()
plt.show()


train_acc = [t/600 for t in train_correct]
test_acc = [t/100 for t in test_correct]

test_correct


plt.plot(train_acc, label='Training Accuracy')
plt.plot(test_acc, label='Test Accuracy')
plt.legend()
plt.show()










