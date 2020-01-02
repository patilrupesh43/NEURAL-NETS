#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 19:30:43 2019

@author: rupesh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    
    def __init__(self, in_features = 4, hidden_layer_1 = 8, hidden_layer_2 = 9, out_features = 3):
        
        super().__init__()
        
        self.fc1 = nn.Linear(in_features, hidden_layer_1)
        self.fc2 = nn.Linear(hidden_layer_1,hidden_layer_2)
        self.out = nn.Linear(hidden_layer_2, out_features)
        
  
        
        
    def forward(self, X):
        
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.out(X))
        
        return X
        
        
torch.manual_seel = 32
model  = Model()


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Applications/Queens MMA/My Code/Pytorch/PYTORCH_NOTEBOOKS/Data/iris.csv')

pd.set_option('display.max_columns', None)
df.head()


X = df.drop('target', axis=1).values
y = df['target'].values


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 33)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)


epochs = 100
losses = []


for i in range(epochs):
    i += 1
    
    y_pred = model.forward(X_train)
    
    
    loss = criterion(y_pred, y_train)
    
    losses.append(loss)
    
    if i%10 == 0:
        print(f'Epoch {i} loss is: {loss}')
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
plt.plot(range(epochs), losses)


#VAlidating

with torch.no_grad():
    y_eval = model.forward(X_test)
    loss = criterion(y_eval, y_test)
    
loss

correct  = 0

with torch.no_grad():

    for i, data in enumerate(X_test):
        y_val = model.forward(data)
        
        #print(f' (i+1).) {str(y_val.argmax())}   {y_test[i]}')
        
        if y_val.argmax().item() == y_test[i]:
            correct += 1
            
print(f'We got {correct} correct')




torch.save(model.state_dict(), 'my_iris_model.pt')


new_model = Model()
new_model.load_state_dict(torch.load('my_iris_model.pt'))

new_model.eval()

mystery_iris = torch.tensor([5.6, 3.7, 2.2, 0.5])


with torch.no_grad():
    print(new_model(mystery_iris))
    print(new_model(mystery_iris).argmax().item())
    
