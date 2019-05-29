# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 20:28:07 2019

@author: Or
"""
import torch
import torch.nn.functional as F


class ConvNet(torch.nn.Module):
    def __init__(self,H,W,C,Dout):
        super(ConvNet, self).__init__()
        self.H = H
        self.W = W
        self.C = C
        self.Dout = Dout
        self.Chid = 32
        self.Chid2 = 64
        self.Chid3 = 64
        
        self.conv1 = torch.nn.Conv2d(in_channels=self.C,out_channels=self.Chid,kernel_size=3,stride=1,padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=self.Chid,out_channels=self.Chid2,kernel_size=3,stride=1,padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=self.Chid2,out_channels=self.Chid3,kernel_size=3,stride=1,padding=1)
        self.fc1 = torch.nn.Linear(int(self.Chid3*H*W/16),564)
        self.fc2 = torch.nn.Linear(564,Dout)
        
    def forward(self,x):
        batch_size = x.shape[0]
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(F.relu(self.conv3(x)),2)
        x = x.view(batch_size,int(self.Chid3*self.H*self.W/16))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class ConvNet2(torch.nn.Module):
    def __init__(self,H,W,C,Dout):
        super(ConvNet2, self).__init__()
        self.H = H
        self.W = W
        self.C = C
        self.Dout = Dout
        self.Chid = 32
        self.Chid2 = 64
        self.Chid3 = 64
        
        self.conv1 = torch.nn.Conv2d(in_channels=self.C,out_channels=self.Chid,kernel_size=3,stride=1,padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=self.Chid,out_channels=self.Chid2,kernel_size=3,stride=1,padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=self.Chid2,out_channels=self.Chid3,kernel_size=3,stride=1,padding=1)
        self.fc1 = torch.nn.Linear(int(self.Chid3*H*W/16),564)
        self.policy = torch.nn.Linear(564,Dout)
        self.value = torch.nn.Linear(564,1)
        
    def forward(self,x):
        batch_size = x.shape[0]
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(F.relu(self.conv3(x)),2)
        x = x.view(batch_size,int(self.Chid3*self.H*self.W/16))
        x = F.relu(self.fc1(x))
        pi = self.policy(x)
        val = self.value(x)
        return pi,val
