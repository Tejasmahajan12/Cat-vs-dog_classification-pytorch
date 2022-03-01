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
        #CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.2),
        #GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=0.2)
        ],p=0.6)
    
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
        #RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1,p=0.2),
        #CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.2),
        #GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=0.2)
        ],p=0.6)
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
'''    
train=DataLoader(Train(path='/home/griffyn-admin/Downloads/data/train/**/**'),batch_size=4)
for image, label in train:
    print(image.shape)
    print(label.shape)
    break
'''
