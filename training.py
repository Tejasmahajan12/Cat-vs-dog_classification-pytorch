import os
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib
import glob
from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
from tqdm import tqdm
import math
from torchvision.transforms import Compose
from albumentations import *
import PIL
from PIL import Image
from dataloader import Train
from dataloader import Val	
from model import ConvNet
#training and validation...

train_loader=DataLoader(Train(path='/home/griffyn-admin/Downloads/data/train/**/**'),batch_size=4)

test_loader=DataLoader(Val(path='/home/griffyn-admin/Downloads/data/test/**/**'),batch_size=4)
print('Length of Train Loader: ',len(train_loader))
print('Length of Test Loader: ',len(test_loader))

cuda=torch.cuda.is_available()

model=ConvNet(num_classes=2)


if cuda:
    model=model.cuda()

#loss and optimizer..
my_loss=nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)



train_loss=[]
train_accuracy=[]
epochs=10
val_loss=[]
val_accuracy=[]
iterations=0
val_itererations=0
best_accuracy=0.0
for epochs in range(epochs):
    start=time.time()
    correct=0
    iter_loss=0
    model.train()
    
    for i,(image,label) in tqdm(enumerate(train_loader)):
        if cuda:
            image=image.cuda()
            label=label.cuda()
        #print('before squeze:',label)
        #print('before squeze:',label.shape)
        label=label.squeeze()
        if image.shape[0]==1:
            label=label[None]
        
        #print('after squeze:',label)
        #print('after squeze:',label.shape)
        
        
        #print('shape of output',output)
        #print('shape of label',label)
        output=model(image)
        if cuda:
            loss = my_loss(output,label)
        else:
            loss = my_loss(output,label).cpu()
        #print(' loooooooss:',loss.item())
        optimizer.zero_grad()
        iter_loss +=loss.data
        
       
        loss.backward()
        optimizer.step()

        _,pred=torch.max(output,1)
        #print('pred: ',pred)
        correct +=(pred==label).sum()
        iterations +=1
#         print('Iterations: ',iterations)
#         print(model.parameters())
    
        
    train_loss.append(iter_loss/iterations)
    train_accuracy.append(100*correct/float(len(train_loader)*4))
#print('Epochs {},traning Loss: {:.3f},train_accuracy: {:.3f}'.format(epochs,train_loss[-1],train_accuracy[-1]))


    
    val_iter_loss=0
    val_correct=0
    model.eval()
    with torch.no_grad():
        for i, (image,label) in tqdm(enumerate(test_loader)):
            if cuda:
                image=image.cuda()
                label=label.cuda()
            label=label.squeeze_()
            if image.shape[0]==1:
                label=label[None]
        
            output=model(image)
            if cuda:
                loss=my_loss(output,label)
            else:
                loss=my_loss(output,label).cpu()
        
            val_iter_loss +=loss.data

            _,pred=torch.max(output,1)

            val_correct +=(pred==label).sum()
            val_itererations +=1
        
        val_loss.append(val_iter_loss/val_itererations)
        val_accuracy.append(100*val_correct/float(len(test_loader)*image.shape[0]))
        stop = time.time()
    print('Epochs {},traning Loss: {:.3f},train_accuracy: {:.3f},val loss :{:.3f},val accuracy: {:3f}'.format(epochs,train_loss[-1],train_accuracy[-1],val_loss[-1],val_accuracy[-1]))
   
 #Save the best model
    if val_accuracy>=best_accuracy:
        torch.save(model.state_dict(),'model_pytroch')
        best_accuracy=val_accuracy
    
       

