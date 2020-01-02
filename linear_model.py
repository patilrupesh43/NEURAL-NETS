#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 21:52:26 2019

@author: rupesh
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

x = torch.linspace(1,50,50).reshape(-1,1)

torch.manual_seed(71)

e = torch.randint(-8,9,(50,1), dtype = torch.float)

y = 2*x + 1 + e

y.shape

plt.scatter(x.numpy(), y.numpy())
plt.show()

torch.manual_seed(59)

'''
model = nn.Linear(in_features=1, out_features=1)
print(model.weight)
print(model.bias)
'''

class Model(nn.Module):
    
    def __init__(self, in_features, out_features):
        
        super().__init__()
        self.linear = nn.Linear(in_features, out_features) 
                                 
    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred
    

    
torch.manual_seed(59)
model = Model(1,1)

print(model.linear.weight)
print(model.linear.bias)

for name,param in model.named_parameters():
    print(name,'\t', param)
    
'''
x = torch.tensor([2.0])
print(model.forward(x))
'''

x1 = torch.linspace(0,50,50)
x1

w1 = 0.1059
b1 = 0.9637

y1 = w1 * x1 + b1

plt.scatter(x.numpy(), y.numpy())
plt.plot(x1,y1,'r')
plt.show()

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)

epochs = 50
losses =[]

for i in range (epochs):
    i += 1
    
    y_pred = model.forward(x)
    
    loss = criterion(y_pred, y)
    
    losses.append(loss)
    
    print(f"epoch {i} loss: {loss.item()} weight: {model.linear.weight.item()} bias: {model.linear.bias.item()} ")

    optimizer.zero_grad()
    
    loss.backward()
    optimizer.step()
    
    
plt.plot(range(epochs), losses)
plt.ylabel('MSE loss')
plt.xlabel('EPOCH')
    

w1 = model.linear.weight.item()
b1 = model.linear.bias.item()

y1 = w1 * x1 + b1

    
plt.scatter(x.numpy(), y.numpy())
plt.plot(x1,y1,'r')
plt.show()
    

