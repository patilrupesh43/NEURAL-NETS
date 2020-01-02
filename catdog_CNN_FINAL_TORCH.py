#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 12:24:50 2019

@author: rupesh
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import os
from PIL import Image
from IPython.display import display

import warnings
warnings.filterwarnings('ignore')


train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
     ])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
     ])

path = '/Applications/Queens MMA/My Code/Pytorch/catdog/CATS_DOGS/'

train_data = datasets.ImageFolder(os.path.join(path, 'train'),transform = train_transform)
test_data = datasets.ImageFolder(os.path.join(path, 'test'),transform = test_transform)

torch.manual_seed(42)

train_loader = DataLoader(train_data, batch_size = 10, shuffle = True)
test_loader = DataLoader(test_data, batch_size=10)

class_name = train_data.classes

class_name

len(train_data)


class ConvolutionalNetwork(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 6, 3 ,1)
        self.conv2 = nn.Conv2d(6,16,3,1)
        self.fc1 = nn.Linear(54*54*16, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,2)
        
    def forward(self, X):
        
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2,2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2,2)
        X = X.view(-1, 54*54*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim =1)
    
    
torch.manual_seed((101))

model = ConvolutionalNetwork()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr =0.001)    

for p in model.parameters():
    print(p.numel())


import time

start_time = time.time()

epochs = 3

#LIMIT NO of BATCHES
max_trn_batch = 800 #Each batch has 10 images so total 800 images
max_tst_batch = 300 #Each batch has 10 images so total 300 images

train_loss = []
test_loss = []
train_corr = []
test_corr = []

for i in range(epochs):
    
    trn_corr = 0
    tst_corr = 0
    
    for b, (X_train,y_train) in enumerate(train_loader):
        if b == max_trn_batch:
            break
        
        b += 1
        
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        
        predicted = torch.max(y_pred.data,1)[1]
        batch_correct = (predicted == y_train).sum()
        trn_corr += batch_correct
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if b % 200 == 0:
            print(f'Epoch {i} batch {b} LOSS: {loss.item()} ' )
            
    train_loss.append(loss)
    train_corr.append(trn_corr)
    
    with torch.no_grad():
        
        for b, (X_test,y_test) in enumerate(test_loader):
            
            if b == max_tst_batch:
                break
            
            y_val = model(X_test)
            
            predicted = torch.max(y_val.data,1)[1]
            batch_correct = (predicted == y_test).sum()
            tst_corr += batch_correct
    
    loss = criterion(y_val, y_test)
    test_loss.append(loss)
    test_corr.append(tst_corr)
            

total_time = time.time() - start_time
print(f'total time {total_time/60} minutes')
    
    


    
torch.save(model.state_dict(),'/Applications/Queens MMA/My Code/Pytorch/catdog/CNN_catdog.pt')


plt.plot(train_loss, label = 'TRaining Loss')
plt.plot(test_loss, label = 'Test Loss')
plt.title('Loss at Epochs')
plt.legend()
plt.show()

plt.plot([t/80 for t in train_corr], label = 'Training Accuracy')
plt.plot([t/30 for t in test_corr], label = 'Test Accuracy')
plt.title('Accuracy at Epochs')
plt.legend()
plt.show()  
    

100*(test_corr[-1].item() / 3000 )











AlexNetModel = models.alexnet(pretrained = True)
AlexNetModel

for param in AlexNetModel.parameters():
    param.requires_grad = False
    
    
torch.manual_seed(42)

AlexNetModel.classifier = nn.Sequential(nn.Linear(9216,1024),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(1024,2),
                                        nn.LogSoftmax(dim=1)
                                        )    

AlexNetModel

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(AlexNetModel.classifier.parameters(), lr=0.01)    
    
    
import time

start_time = time.time()

epochs = 1

#LIMIT NO of BATCHES
max_trn_batch = 800 #Each batch has 10 images so total 800 images
max_tst_batch = 300 #Each batch has 10 images so total 300 images

train_loss = []
test_loss = []
train_corr = []
test_corr = []

for i in range(epochs):
    
    trn_corr = 0
    tst_corr = 0
    
    for b, (X_train,y_train) in enumerate(train_loader):
        if b == max_trn_batch:
            break
        
        b += 1
        
        y_pred = AlexNetModel(X_train)
        loss = criterion(y_pred, y_train)
        
        predicted = torch.max(y_pred.data,1)[1]
        batch_correct = (predicted == y_train).sum()
        trn_corr += batch_correct
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if b % 200 == 0:
            print(f'Epoch {i} batch {b} LOSS: {loss.item()} ' )
            
    train_loss.append(loss)
    train_corr.append(trn_corr)
    
    with torch.no_grad():
        
        for b, (X_test,y_test) in enumerate(test_loader):
            
            if b == max_tst_batch:
                break
            
            y_val = AlexNetModel(X_test)
            
            predicted = torch.max(y_val.data,1)[1]
            batch_correct = (predicted == y_test).sum()
            tst_corr += batch_correct
    
    loss = criterion(y_val, y_test)
    test_loss.append(loss)
    test_corr.append(tst_corr)
            

total_time = time.time() - start_time
print(f'total time {total_time/60} minutes')
    
test_corr
print(test_corr[-1].item()/3000)






    
model = ConvolutionalNetwork()
model.load_state_dict(torch.load('/Applications/Queens MMA/My Code/Pytorch/catdog/CNN_catdog.pt'))
model.eval()


demo_image = Image.open('/Applications/Queens MMA/My Code/Pytorch/catdog/rand_test2.jpg')
display(demo_image)
demo_im = train_transform(demo_image)

with torch.no_grad():
    new_pred =  model(demo_im.view(1,3,224,224)).argmax()
    
class_name[new_pred.item()]


with torch.no_grad():
    new_pred =  AlexNetModel(demo_im.view(1,3,224,224)).argmax()
    
class_name[new_pred.item()]
