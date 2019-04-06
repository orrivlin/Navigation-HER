# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 23:23:01 2019

@author: Or
"""

import torch
import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from smooth_signal import smooth
from Nav2D import Navigate2D
from Models import ConvNet
import time
from dqn_HER import DQN_HER
from copy import deepcopy as dc
import time
from collections import deque


N = 20
Nobs = 15
Dobs = 2
Rmin = 10
env = Navigate2D(N,Nobs,Dobs,Rmin)
gamma = 0.99
buffer_size = 1000000
ddqn = True
alg = DQN_HER(env, gamma, buffer_size, ddqn)
epochs = 15000

distances = deque(maxlen=100)

n_data = 400
for i in range(n_data):
    min_dist = alg.gather_data()  
    distances.append(min_dist)
alg.calc_norm()
for i in range(epochs):
    T1 = time.time()
    log = alg.run_epoch()
    T2 = time.time()
    distances.append(log.get_current('final_dist'))
    Y = np.asarray(distances)
    Y[Y > 1] = 1.0
    Y = 1 - Y
    print('done: {} of {}. loss: {}. success rate: {}. time: {}'.format(i,epochs,np.round(log.get_current('avg_loss'),2),np.round(np.mean(Y),2),np.round(T2-T1,3)))
    if (i % 100) == 0:
        torch.save(alg.model.state_dict(),'model.pt')
        torch.save(alg.image_mean, 'norm.pt')


Y = np.asarray(log.get_log('final_dist'))
Y2 = smooth(Y)
x = np.linspace(0, len(Y), len(Y))
fig1 = plt.figure()
ax1 = plt.axes()
ax1.plot(x, Y, Y2)
plt.xlabel('episodes')
plt.ylabel('minimum episode distance')

Y = np.asarray(log.get_log('avg_loss'))
Y2 = smooth(Y)
x = np.linspace(0, len(Y), len(Y))
fig2 = plt.figure()
ax2 = plt.axes()
ax2.plot(x, Y, Y2)
plt.xlabel('episodes')
plt.ylabel('average loss')

Y = np.asarray(log.get_log('tot_return'))
Y2 = smooth(Y)
x = np.linspace(0, len(Y), len(Y))
fig3 = plt.figure()
ax3 = plt.axes()
ax3.plot(x, Y, Y2)
plt.xlabel('episodes')
plt.ylabel('episode return')


Y = np.asarray(log.get_log('final_dist'))
Y[Y > 1] = 1.0
Y = 1 - Y
K = 100
Y2 = smooth(Y,window_len=K)
x = np.linspace(0, len(Y2), len(Y2))
fig3 = plt.figure()
ax3 = plt.axes()
ax3.plot(x,Y2)
plt.xlabel('episodes')
plt.ylabel('success rate')
