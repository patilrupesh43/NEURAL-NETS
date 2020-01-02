#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 20:44:00 2019

@author: rupesh
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('/Applications/Queens MMA/My Code/Pytorch/PYTORCH_NOTEBOOKS/Data/NYCTaxiFares.csv')

pd.set_option('max_columns', 500)
df.head()

df['fare_amount'].describe()



def haversine_distance(df, lat1, long1, lat2, long2):
    """
    Calculates the haversine distance between 2 sets of GPS coordinates in df
    """
    r = 6371  # average radius of Earth in kilometers
       
    phi1 = np.radians(df[lat1])
    phi2 = np.radians(df[lat2])
    
    delta_phi = np.radians(df[lat2]-df[lat1])
    delta_lambda = np.radians(df[long2]-df[long1])
     
    a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = (r * c) # in kilometers

    return d

df.columns


df['distance'] = haversine_distance(df, 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')


df.info()

df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])


my_time = df['pickup_datetime'][0]


df['EDTdate'] = df['pickup_datetime'] - pd.Timedelta(hours = 4)
df['hour'] = df['EDTdate'].dt.hour
df['AMorPM'] = np.where( df['hour']< 12, 'am', 'pm')
df['Weekday'] = df['EDTdate'].dt.strftime("%a")

cat_cols = ['hour', 'AMorPM', 'Weekday']
for cat in cat_cols:
    df[cat] = df[cat].astype('category')
cats = np.stack([ df[cat].cat.codes.values for cat in cat_cols], axis = 1)
cats = torch.tensor(cats, dtype = torch.int64)

'''
df['hour'].head()

df['AMorPM'].cat.categories
df['AMorPM'].cat.codes

df['Weekday'].cat.categories
df['Weekday'].cat.codes


hr = df['hour'].cat.codes.values  #convert to numpy array
ampm = df['AMorPM'].cat.codes.values 
wkdy = df['Weekday'].cat.codes.values 
cats = np.stack([hr,ampm,wkdy])
'''

cont_cols =['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude','passenger_count', 'distance']
conts = np.stack([ df[cont].values for cont in cont_cols], 1)
conts = torch.tensor(conts, dtype = torch.float)
conts


y_col = df['fare_amount']
y = torch.tensor(y_col, dtype  = torch.float).reshape(-1,1)


cats.shape
conts.shape
y.shape


cat_siz = [len(df[col].cat.categories) for col in cat_cols]
cat_siz

emb_siz = [ (size, min(50, (size+1)//2)) for size in cat_siz]
emb_siz





'''
catz = cats[:4]

selfembeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_siz])
selfembeds

embeddings = []
for i, e in enumerate(selfembeds):
    embeddings.append(e(catz[:,i]))
    
embeddings

'''


class TabularModel(nn.Module):
    
    
    def __init__(self,emb_siz, n_cont, out_sz, layers, p=0.5):
        
        super().__init__()
        
        self.embeds = nn.ModuleList([nn.Embedding(ni,nf) for ni,nf in emb_siz])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        
        layerlist = []
        n_emb = sum((nf for ni, nf in emb_siz))
        n_in = n_emb + n_cont
        
        
        for i in layers:
            layerlist.append(nn.Linear(n_in, i))
            layerlist.append(nn.ReLU(inplace = True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        
        layerlist.append(nn.Linear(layers[-1], out_sz))
        
        self.layers = nn.Sequential(*layerlist)
        
    
    def forward(self, x_cat, x_cont):
        
        embeddings = []
        
        for i, e in enumerate(self.embeds):
                embeddings.append(e(x_cat[:,i]))
        
                
        x  = torch.cat(embeddings,1)
        x  = self.emb_drop(x)    
        
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        
        x = self.layers(x)
        
        return x


torch.manual_seed(33)
        
        
    
model = TabularModel(emb_siz, conts.shape[1], 1, [200,100], 0.4)
model
            
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

batch_size = 120000

test_size = int(batch_size*0.2)            

cat_train = cats[:batch_size-test_size]
cat_test = cats[batch_size-test_size : batch_size]
con_train = conts[:batch_size-test_size]
con_test  = conts[batch_size-test_size : batch_size]      
y_train = y[:batch_size-test_size]
y_test =  y[batch_size-test_size : batch_size]


cat_train.shape

import time

start_time = time.time()

epochs = 150

losses = []

for i in range(epochs):
    i = i+1
    
    y_pred = model.forward(cat_train, con_train)
    loss = torch.sqrt(criterion(y_pred,y_train))          
    
    losses.append(loss)
    
    if i%10 ==1:
        print(f'epoch {i} loss: {loss}')
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
duration = time.time() - start_time
print(f'time elapsed: {duration/60} minutes')
        
y_pred.shape
'''

plt.plot(range(epochs), losses)
plt.xlabel('epoch')
plt.ylabel('ylabel')
plt.show()


with torch.no_grad():
    y_val = model(cat_test,con_test)
    loss = torch.sqrt(criterion(y_val,y_test))
    
loss

for i in range(10):
    print(f'{i} PREDICTED {y_val[i].item():8.2f} TRUE: {y_test[i].item():8.2f}')
    
    
torch.save(model.state_dict(),'TaxiModel.pt')


det_model = TabularModel(emb_siz, conts.shape[1], 1, [200,100], 0.4)

det_model.load_state_dict(torch.load('/Applications/Queens MMA/My Code/Pytorch/TaxiModel.pt'))

det_model

'''

det_model = model










df_test = pd.read_csv('/Applications/Queens MMA/Kaggle/Newyorktaxi fare price/test.csv')
df_test.info()
key_d = df_test['key']
df_test = df_test.drop(['key'], axis = 1)

df_test['distance'] = haversine_distance(df_test, 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')
df_test['EDTdate'] = df['pickup_datetime'] - pd.Timedelta(hours = 4)
df_test['hour'] = df['EDTdate'].dt.hour
df_test['AMorPM'] = np.where( df_test['hour']< 12, 'am', 'pm')
df_test['Weekday'] = df['EDTdate'].dt.strftime("%a")

df_test.info()

test_cat_cols = ['hour', 'AMorPM', 'Weekday']
for cat in test_cat_cols:
    df_test[cat] = df_test[cat].astype('category')
test_cats = np.stack([ df_test[cat].cat.codes.values for cat in test_cat_cols], axis = 1)
test_cats = torch.tensor(test_cats, dtype = torch.int64)

test_cont_cols =['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude','passenger_count', 'distance']
test_conts = np.stack([ df_test[cont].values for cont in test_cont_cols], 1)
test_conts = torch.tensor(test_conts, dtype = torch.float)


with torch.no_grad():
    y_val = model(test_cats,test_conts)
    #loss = torch.sqrt(criterion(y_val,y_test))
    
for i in range(100):
    print(f'{i} PREDICTED {y_val[i].item()}')

holder = pd.DataFrame()
holder['key'] = key_d
#demo['fare'] = pd.DataFrame(y_val.data.cpu().numpy())
holder['fare_amount'] = pd.DataFrame(y_val.data.cpu().numpy())

holder.to_csv('/Applications/Queens MMA/Kaggle/Newyorktaxi fare price/submission.csv', index = False)



