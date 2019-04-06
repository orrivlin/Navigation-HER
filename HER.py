# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 23:42:57 2019

@author: Or
"""
from collections import deque
import torch 
import numpy as np
import copy


class HER:
    def __init__(self):
        self.buffer = deque()
        
    def reset(self):
        self.buffer = deque()
        
    def keep(self,item):
        self.buffer.append(item)
        
    def backward(self):
        num = len(self.buffer)
        goal = self.buffer[-1][-2][1,:,:]
        for i in range(num):
            self.buffer[-1-i][-2][2,:,:] = goal
            self.buffer[-1-i][0][2,:,:] = goal
            self.buffer[-1-i][2] = -1.0
            self.buffer[-1-i][4] = False
            if np.sum(np.abs(self.buffer[-1-i][-2][1,:,:] - goal)) == 0:
                self.buffer[-1-i][2] = 0.0
                self.buffer[-1-i][4] = True
        return self.buffer
        
        