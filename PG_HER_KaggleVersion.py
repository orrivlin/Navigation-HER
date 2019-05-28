

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from copy import deepcopy as dc
import torch.nn.functional as F
from collections import deque
import time


def smooth(x,window_len=11,window='hanning'):
    if window_len<3:
        return x

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

class mean_val:
    def __init__(self):
        self.k = 0
        self.val = 0
        self.mean = 0
        
    def append(self,x):
        self.k += 1
        self.val += x
        self.mean = self.val/self.k
        
    def get(self):
        return self.mean
        
    
class logger:
    def __init__(self):
        self.log = dict()
        
    def add_log(self,name):
        self.log[name] = []
        
    def add_item(self,name,x):
        self.log[name].append(x)
        
    def get_log(self,name):
        return self.log[name]
    
    def get_keys(self):
        return self.log.keys()
    
    def get_current(self,name):
        return self.log[name][-1]

class Navigate2D:
    def __init__(self,N,Nobs,Dobs,Rmin):
        self.N = N
        self.Nobs = Nobs
        self.Dobs = Dobs
        self.Rmin = Rmin
        self.state_dim = [N,N,3]
        self.action_dim = 4
        self.scale = 10.0
        
    def get_dims(self):
        return self.state_dim, self.action_dim
        
    def reset(self):
        grid = np.zeros((self.N,self.N,3))
        for i in range(self.Nobs):
            center = np.random.randint(0,self.N,(1,2))
            minX = np.maximum(center[0,0] - self.Dobs,1)
            minY = np.maximum(center[0,1] - self.Dobs,1)
            maxX = np.minimum(center[0,0] + self.Dobs,self.N-1)
            maxY = np.minimum(center[0,1] + self.Dobs,self.N-1)
            grid[minX:maxX,minY:maxY,0] = 1.0
            
        free_idx = np.argwhere(grid[:,:,0] == 0.0)
        start = free_idx[np.random.randint(0,free_idx.shape[0],1),:].squeeze()
        while (True):
            finish = free_idx[np.random.randint(0,free_idx.shape[0],1),:].squeeze()
            if ((start[0] != finish[0]) and (start[1] != finish[1]) and (np.linalg.norm(start - finish) >= self.Rmin)):
                break
        grid[start[0],start[1],1] = self.scale*1.0
        grid[finish[0],finish[1],2] = self.scale*1.0
        done = False
        return grid, done
    
    def step(self,grid,action):        
        new_grid = dc(grid)
        done = False
        reward = 0.0
        act = np.array([[1,0],[0,1],[-1,0],[0,-1]])
        pos = np.argwhere(grid[:,:,1] == self.scale**1.0)[0]
        target = np.argwhere(grid[:,:,2] == self.scale*1.0)[0]
        new_pos = pos + act[action]
        
        dist2 = np.linalg.norm(new_pos - target)
        if (np.any(new_pos < 0.0) or np.any(new_pos > (self.N - 1)) or (grid[new_pos[0],new_pos[1],0] == 1.0)):
            return grid, reward, done, dist2
        new_grid[pos[0],pos[1],1] = 0.0
        new_grid[new_pos[0],new_pos[1],1] = self.scale*1.0
        if ((new_pos[0] == target[0]) and (new_pos[1] == target[1])):
            reward = 10.0
            done = True
        return new_grid, reward, done, dist2
    
    def get_tensor(self,grid):
        S = torch.Tensor(grid).transpose(2,1).transpose(1,0).unsqueeze(0)
        return S
    
    def render(self,grid):
        plot = imshow(grid)
        return plot
    

    
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


class HER:
    def __init__(self,N,cuda_flag):
        self.buffer = deque()
        self.N = N
        self.cuda = cuda_flag
        
    def reset(self):
        self.buffer = deque()
        
    def keep(self,item):
        self.buffer.append(item)
        
    def backward(self,model,gamma):
        K = len(self.buffer)
        new_buffer = deque()
        for i in range(K):
            new_buffer.append(self.buffer[i])
        num = len(new_buffer)
        goal = new_buffer[-1][2][0,1,:,:]
        for i in range(num):
            new_buffer[-1-i][3] = 10.0
            new_buffer[-1-i][0][0,2,:,:] = goal
            if self.cuda:
                [pi,v] = model(new_buffer[-1-i][0].cuda())
                new_buffer[-1-i][1] = F.softmax(pi,dim=1).squeeze()[new_buffer[-1-i][1]]
            else:
                [pi,v] = model(new_buffer[-1-i][0])
                new_buffer[-1-i][1] = F.softmax(pi,dim=1).squeeze()[new_buffer[-1-i][1]]
            if ((new_buffer[-1-i][2][0,1,:,:] - goal).abs().sum() == 0):
                new_buffer[-1-i][3] = 10.0
        for t in range(len(new_buffer)):
            if (t==0):
                X = new_buffer[t][0]
                ratio = new_buffer[t][1].detach().item()/new_buffer[t][4]
                PI = ratio*new_buffer[t][1].unsqueeze(0)
                R = torch.Tensor(np.array(new_buffer[t][3])).unsqueeze(0)
                V = v.unsqueeze(0)
            else:
                X = torch.cat([X,new_buffer[t][0]],dim=0)
                ratio = new_buffer[t][1].detach().item()/new_buffer[t][4]
                PI = torch.cat([PI,ratio*new_buffer[t][1].unsqueeze(0)],dim=0)
                R = torch.cat([R,torch.Tensor(np.array(new_buffer[t][3])).unsqueeze(0)],dim=0)
                V = torch.cat([V,v.unsqueeze(0)])
        return X,PI,R,V


class DiscretePolicyGradient:
    def __init__(self, env, gamma, num_episodes,cuda_flag):
        self.env = env
        [Sdim,Adim] = env.get_dims()
        if cuda_flag:
            self.model = ConvNet(Sdim[0],Sdim[0],3,Adim).cuda()
        else:
            self.model = ConvNet(Sdim[0],Sdim[0],3,Adim)
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.0003)
        self.batch_size = 32
        self.epsilon = 1e-8
        self.num_episodes = num_episodes
        self.her = HER(self.env.N,cuda_flag)
        self.cuda = cuda_flag
        self.log = logger()
        self.log.add_log('reward')
        self.log.add_log('final_dist')
        self.log.add_log('TD_error')
        self.log.add_log('entropy')
        
    def run_episode(self):
        obs, done = self.env.reset()
        state = self.env.get_tensor(obs)
        sum_r = 0
        min_dist = self.env.N
        self.her.reset()
        max_time = 50
        for t in range(max_time):
            if self.cuda:
                [pi,val] = self.model(state.cuda())
            else:
                [pi,val] = self.model(state)
            pi = F.softmax(pi,dim=1)
            dist = torch.distributions.categorical.Categorical(pi.squeeze())
            action = dist.sample().item()
            new_obs, reward, done, dist = self.env.step(obs,action)
            obs = dc(new_obs)
            new_state = self.env.get_tensor(obs)
            sum_r = sum_r + reward
            if dist < min_dist:
                min_dist = dist
            
            self.her.keep([dc(state),dc(action),dc(new_state),dc(reward),dc(pi[0,action].detach().item())])
   
            if (t==0):
                X = state
                PI = pi[0,action].unsqueeze(0)
                R = torch.Tensor(np.array(reward)).unsqueeze(0)
                V = val.unsqueeze(0)
            else:
                X = torch.cat([X,state],dim=0)
                PI = torch.cat([PI,pi[0,action].unsqueeze(0)],dim=0)
                R = torch.cat([R,torch.Tensor(np.array(reward)).unsqueeze(0)],dim=0)
                V = torch.cat([V,val.unsqueeze(0)],dim=0)
            state = new_state
            if done:
                break
        self.log.add_item('reward',sum_r)
        self.log.add_item('final_dist',min_dist)
        tot_return = R.sum().item()
        for i in range(R.shape[0] - 1):
            R[-2-i] = R[-1]
        [XX,PIPI,RR,VV] = self.her.backward(self.model,self.gamma)
        X = torch.cat((X,XX),dim=0)
        PI = torch.cat((PI,PIPI),dim=0)
        R = torch.cat((R,RR),dim=0)
        V = torch.cat((V,VV),dim=0)

        
        return X, PI, R, V, tot_return
    
    
    def update_model(self,PI,R,V):
        self.optimizer.zero_grad()
        if self.cuda:
            R = R.cuda()
        A = R.squeeze() - V.squeeze().detach()
        #A = R
        L_policy = -(torch.log(PI)*A).mean()
        L_value = F.smooth_l1_loss(V.squeeze(), R.squeeze())
        L_entropy = -(PI*PI.log()).mean()
        L = L_policy + L_value - 0.01*L_entropy
        L.backward()
        self.optimizer.step()
        self.log.add_item('TD_error',L_value.detach().item())
        self.log.add_item('entropy',L_entropy.detach().item())

    
    def run_epoch(self):
        mean_return = 0
        for i in range(self.num_episodes):
            [x,pi,r,val,tot_return] = self.run_episode()
            mean_return = mean_return + tot_return
            if (i == 0):
                PI = pi
                R = r
                V = val
            else:
                PI = torch.cat([PI,pi],dim=0)
                R = torch.cat([R,r],dim=0)
                V = torch.cat([V,val],dim=0)
                
        mean_return = mean_return/self.num_episodes
        self.update_model(PI,R,V)
        return self.log


N = 20
Nobs = 15
Dobs = 2
Rmin = 10
env = Navigate2D(N,Nobs,Dobs,Rmin)

gamma = 0.99
num_episodes = 10
alg = DiscretePolicyGradient(env,gamma,num_episodes,True)
num_epochs = 400

for i in range(num_epochs):
    mean_time = mean_val()
    success_rate = mean_val()
    mean_loss = mean_val()
    mean_h = mean_val()
    for j in range(100):
        T1 = time.time()
        log = alg.run_epoch()
        T2 = time.time()
        mean_time.append(T2-T1)
        mean_loss.append(log.get_current('TD_error'))
        mean_h.append(log.get_current('entropy'))
        if log.get_current('final_dist') > 0.0:
            success_rate.append(0.0)
        else:
            success_rate.append(1.0)
        if (j % 10) == 0:
            torch.save(alg.model.state_dict(),'nav2d_model_PG.pt')

    print('Done: {} of {}. TD error: {}. success rate: {}. entropy: {}. mean iteration time: {}'.format(i*100,num_epochs*100,np.round(mean_loss.get(),2),np.round(success_rate.get(),2),np.round(mean_h.get(),2),np.round(mean_time.get(),3)))



tot_ret_i = log.get_log('reward')
Y = np.asarray(tot_ret_i)
Y2 = smooth(Y)
x = np.linspace(0, len(Y), len(Y))
fig1 = plt.figure()
ax1 = plt.axes()
ax1.plot(x, Y, Y2)
plt.xlabel('episodes')
plt.ylabel('episode return')

tot_ret_i = log.get_log('final_dist')
Y = np.asarray(tot_ret_i)
Y2 = smooth(Y)
x = np.linspace(0, len(Y), len(Y))
fig2 = plt.figure()
ax2 = plt.axes()
ax2.plot(x, Y, Y2)
plt.xlabel('episodes')
plt.ylabel('episode final distance')

Y = np.asarray(log.get_log('final_dist'))
Y[Y > 1] = 1.0
Y = 1 - Y
K = 1000
Y2 = smooth(Y,window_len=K)
x = np.linspace(0, len(Y2), len(Y2))
fig3 = plt.figure()
ax3 = plt.axes()
ax3.plot(x,Y2)
plt.xlabel('episodes')
plt.ylabel('success rate')