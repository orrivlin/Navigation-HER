# Navigation-HER
2D navigation using DQN and Hindsight Experience Replay

This repository contains a PyTorch implementation of a simple 2D navigation environment, in which an agent needs to traverse a map and arrive at a destination pixel, while circumventing onstacles. Both agent position and goal are given implicitly in the input image.
For every step in which the agent has not arrived at the goal, it recieves a -1 reward, which makes the problem difficult. To train the agent, I am using a standard DQN algorithm coupled with HER (Hindsight Experience Replay), which helps to overcome the sparse rewards. This is a work in progress, so far I have only managed to achieve around 80% success rate in arriving at the goal, and training takes quite a few hours. Hopefully in the future I can achieve higher success rates. I tried toying with the size of the neural network, but did not see much improvement when using larger models.
