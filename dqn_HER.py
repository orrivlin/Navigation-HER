
"""
@author: orrivlin
"""

import torch 
import numpy as np
import copy
import torch.nn.functional as F
from collections import deque
from Models import ConvNet
import random
from log_utils import logger, mean_val
from HER import HER
from copy import deepcopy as dc



class DQN_HER:
    def __init__(self, env, gamma, buffer_size, ddqn):
        self.env = env
        [Sdim,Adim] = env.get_dims()
        self.model = ConvNet(Sdim[0],Sdim[0],3,Adim).cuda()
        self.target_model = copy.deepcopy(self.model).cuda()
        self.her = HER()
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.0001)
        self.batch_size = 16
        self.epsilon = 0.1
        self.buffer_size = buffer_size
        self.step_counter = 0
        self.epsi_high = 0.9
        self.epsi_low = 0.1
        self.steps = 0
        self.count = 0
        self.decay = 2000
        self.eps = self.epsi_high
        self.update_target_step = 3000
        self.log = logger()
        self.log.add_log('tot_return')
        self.log.add_log('avg_loss')
        self.log.add_log('final_dist')
        self.log.add_log('buffer')
        self.image_mean = 0
        self.image_std = 0
        self.ddqn = ddqn
        
        self.replay_buffer = deque(maxlen=buffer_size)
        
    def run_episode(self):
        self.her.reset()
        obs, done = self.env.reset()
        done = False
        state = self.env.get_tensor(obs)
        sum_r = 0
        mean_loss = mean_val()
        min_dist = 100000
        max_t = 50

        for t in range(max_t):
            self.steps += 1
            self.eps = self.epsi_low + (self.epsi_high-self.epsi_low) * (np.exp(-1.0 * self.steps/self.decay))
            Q = self.model(self.norm(state.cuda()))
            num = np.random.rand()
            if (num < self.eps):
                action = torch.randint(0,Q.shape[1],(1,)).type(torch.LongTensor)
            else:
                action = torch.argmax(Q,dim=1)
            new_obs, reward, done, dist = self.env.step(obs,action.item())
            new_state = self.env.get_tensor(new_obs)
            sum_r = sum_r + reward
            if dist < min_dist:
                min_dist = dist
            if (t+1) == max_t:
                done = True
            
            self.replay_buffer.append([dc(state.squeeze(0).numpy()),dc(action),dc(reward),dc(new_state.squeeze(0).numpy()),dc(done)])
            self.her.keep([state.squeeze(0).numpy(),action,reward,new_state.squeeze(0).numpy(),done])
            loss = self.update_model()
            mean_loss.append(loss)
            state = dc(new_state)
            obs = dc(new_obs)
            
            self.step_counter = self.step_counter + 1
            if (self.step_counter > self.update_target_step):
                self.target_model.load_state_dict(self.model.state_dict())
                self.step_counter = 0
                print('updated target model')
        her_list = self.her.backward()
        for item in her_list:
            self.replay_buffer.append(item)
        self.log.add_item('tot_return',sum_r)
        self.log.add_item('avg_loss',mean_loss.get())
        self.log.add_item('final_dist',min_dist)
        
    def gather_data(self):
        self.her.reset()
        obs, done = self.env.reset()
        done = False
        state = self.env.get_tensor(obs)
        sum_r = 0
        min_dist = 100000
        max_t = 50

        for t in range(max_t):
            self.eps = 1.0
            Q = self.model(state.cuda())
            num = np.random.rand()
            if (num < self.eps):
                action = torch.randint(0,Q.shape[1],(1,)).type(torch.LongTensor)
            else:
                action = torch.argmax(Q,dim=1)
            new_obs, reward, done, dist = self.env.step(obs,action.item())
            new_state = self.env.get_tensor(new_obs)
            sum_r = sum_r + reward
            if dist < min_dist:
                min_dist = dist
            if (t+1) == max_t:
                done = True
            
            self.replay_buffer.append([dc(state.squeeze(0).numpy()),dc(action),dc(reward),dc(new_state.squeeze(0).numpy()),dc(done)])
            state = dc(new_state)
            obs = dc(new_obs)
        return min_dist

    def calc_norm(self):
        S0, A0, R1, S1, D1 = zip(*self.replay_buffer)
        S0 = torch.tensor( S0, dtype=torch.float)
        self.image_mean = S0.mean(dim=0).cuda()
        self.image_std = S0.std(dim=0).cuda()
        
    def norm(self,state):
        return state - self.image_mean
        
    def update_model(self):
        self.optimizer.zero_grad()
        num = len(self.replay_buffer)
        K = np.min([num,self.batch_size])
        samples = random.sample(self.replay_buffer, K)
        
        S0, A0, R1, S1, D1 = zip(*samples)
        S0 = torch.tensor( S0, dtype=torch.float)
        A0 = torch.tensor( A0, dtype=torch.long).view(K, -1)
        R1 = torch.tensor( R1, dtype=torch.float).view(K, -1)
        S1 = torch.tensor( S1, dtype=torch.float)
        D1 = torch.tensor( D1, dtype=torch.float)
        
        S0 = self.norm(S0.cuda())
        S1 = self.norm(S1.cuda())
        if self.ddqn == True:
            model_next_acts = self.model(S1).detach().max(dim=1)[1]
            target_q = R1.squeeze().cuda() + self.gamma*self.target_model(S1).gather(1,model_next_acts.unsqueeze(1)).squeeze()*(1 - D1.cuda())
        else:
            target_q = R1.squeeze().cuda() + self.gamma*self.target_model(S1).max(dim=1)[0].detach()*(1 - D1.cuda())
        policy_q = self.model(S0).gather(1,A0.cuda())
        L = F.smooth_l1_loss(policy_q.squeeze(),target_q.squeeze())
        L.backward()
        self.optimizer.step()
        return L.detach().item()
    
    def run_epoch(self):
        self.run_episode()
        self.log.add_item('buffer',len(self.replay_buffer))
        return self.log

