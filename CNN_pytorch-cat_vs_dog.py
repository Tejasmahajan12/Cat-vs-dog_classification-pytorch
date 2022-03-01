#!/usr/bin/env python
# coding: utf-8

# In[12]:


#Required libraries...
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


# In[13]:


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# In[14]:


#dataloaders...
class Train(Dataset):
    def __init__(self,path='/home/griffyn-admin/Downloads/data/train',apply_aug=False,dim=(100,100)):
        self.dim=dim
        self.img_list=glob.glob('/home/griffyn-admin/Downloads/data/train/**/**')#[:100]
        #self.label=[i.split('/')[-2] for i in self.img_list]
        self.dict={"cat":0,"dog":1}
        self.apply_aug = apply_aug
        self.augmentation=Compose([Rotate(limit = (1.,10.),p=0.5),Flip(0.5),
        RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1,p=0.2),
       
      
    
    def augment(self,img):
        aug_img=self.augmentation(image=img)
        return aug_img['image'] 
        
    def __getitem__(self,index):
        img=self.img_list[index]
        labels=img.split('/')[-2]
        img=Image.open(img)
        img = img.resize(self.dim)
        img=np.asarray(img)/255
        
        if self.apply_aug:
            img=self.augment(img)
        
        labels=self.dict[labels]
        return torch.FloatTensor(img.transpose(2,0,1)),torch.LongTensor([labels])
    
    def __len__(self):
        return len(self.img_list)



class Val(Dataset):
    def __init__(self,path='/home/griffyn-admin/Downloads/data/test',augmentation=True,dim=(100,100)):
        self.dim=dim
        self.img_list=glob.glob('/home/griffyn-admin/Downloads/data/test/**/**')#[:100]
        #self.label=[i.split('/')[-2] for i in self.img_list]
        self.dict={"cat":0,"dog":1}
        self.augmentation=Compose([Rotate(limit = (1.,10.),p=0.5),Flip(0.5),
        
    def augment(self,img):
        aug_img=self.augmentation(image=img)
        return aug_img['image'] 
        
    def __getitem__(self,index):
        img=self.img_list[index]
        labels=img.split('/')[-2]
        img=Image.open(img)
        img = img.resize(self.dim)
        img=np.asarray(img)/255
        
        if self.augmentation:
            img=self.augment(img)
        labels=self.dict[labels]
        return torch.FloatTensor(img.transpose(2,0,1)),torch.LongTensor([labels])
    def __len__(self):
        return len(self.img_list)
    
train=DataLoader(Train(path='/home/griffyn-admin/Downloads/data/train/**/**'),batch_size=4)
for image, label in train:
    print(image.shape)
    print(label.shape)
    break


# In[15]:


#model...

class ConvNet(nn.Module):
    def __init__(self,num_classes=2):
        super(ConvNet,self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
        
        self.bn1=nn.BatchNorm2d(num_features=12)
        self.relu1=nn.ReLU()
        
        self.pool=nn.MaxPool2d(kernel_size=2)
     
        
        self.conv2=nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
        self.relu2=nn.ReLU()
        
        
        
        self.conv3=nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.bn3=nn.BatchNorm2d(num_features=32)
        self.relu3=nn.ReLU()
        
        
        self.linear1=nn.Linear(32,16)
        self.linear2=nn.Linear(16,8)
        self.linear3=nn.Linear(8,2)

        
        
    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)
            
        output=self.pool(output)
            
        output=self.conv2(output)
        output=self.relu2(output)
            
        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)
        #print(output.shape)    
            
            
        output=torch.mean(output.view(output.shape[0],output.shape[1],-1),dim=2)
        #print(output.shape)
        
        output=self.linear1(output)
        output=self.linear2(output)
        output=self.linear3(output)
        #print(output.shape)
            
        
            
        return output
            
#x=torch.rand([4,3,100,100])  
#model=ConvNet()
#model(x)


# In[16]:


model=ConvNet(num_classes=2)
#loss and optimizer...
my_loss=nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1.)


# In[19]:


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

#saving model...
torch.save(model.state_dict(),'best_model.pt')

        
        
    
        


# In[ ]:





# In[ ]:




