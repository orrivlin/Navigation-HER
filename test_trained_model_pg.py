# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 07:57:53 2019

@author: orrivlin
"""

import torch 
import numpy as np
from Models import ConvNet2
from Nav2D import Navigate2D
from copy import deepcopy as dc
import torch.nn.functional as F


N = 20
Nobs = 15
Dobs = 2
Rmin = 10
env = Navigate2D(N,Nobs,Dobs,Rmin)
[Sdim,Adim] = env.get_dims()
model = ConvNet2(Sdim[0],Sdim[0],3,Adim).cuda()
model.load_state_dict(torch.load('nav2d_model_PG.pt'))
model.eval()
start_obs, done = env.reset()
cum_obs = dc(start_obs)
obs = dc(start_obs)
done = False
state = env.get_tensor(obs)
sum_r = 0
epsilon = 0.0
for t in range(50):
    [pi,val] = model(state.cuda())
    num = np.random.rand()
    pi = F.softmax(pi,dim=1)
    dist = torch.distributions.categorical.Categorical(pi.squeeze())
    action = dist.sample().item()
    new_obs, reward, done, dist = env.step(obs,action)
    new_state = env.get_tensor(new_obs)
    sum_r = sum_r + reward
    state = dc(new_state)
    obs = dc(new_obs)
    cum_obs[:,:,1] += obs[:,:,1]
    if done:
        break
env.render(cum_obs)
print('time: {}'.format(t))
print('return: {}'.format(sum_r))
