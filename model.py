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
