# Gridworld Navigation - Hindsight Experience Replay
### 2D navigation using DQN/Actor-Critic and Hindsight Experience Replay

This repository contains a PyTorch implementation of a simple 2D navigation environment, in which an agent needs to traverse a map and arrive at a destination pixel, while circumventing onstacles. Both agent position and goal are given implicitly in the input image.
For every step in which the agent has not arrived at the goal, it recieves a -1 reward, which makes the problem difficult. To train the agent, I started by using a standard DQN algorithm coupled with HER (Hindsight Experience Replay), which helps to overcome the sparse rewards. This has only managed to achieve around 80% success rate in arriving at the goal, and training takes quite a few hours. Next, I implemented an actor-critic version of HER, and recently achieved ~90% success rate in getting to the goal pixel. I think that if I used a more sophisticated learning algorithm such as Proximal-Policy-Optimization or Soft-Actor-Critic, I could probably get better results. This was great fun to work on. I also wrote a Medium article on Hindsight-Experience-Replay, feel free to [check it out](https://towardsdatascience.com/reinforcement-learning-with-hindsight-experience-replay-1fee5704f2f8)

Learning curve for DQN-HER:

![alt text](https://user-images.githubusercontent.com/46422351/55673889-a6f6e000-58b6-11e9-980f-b07cac8b8b13.png)



Learning curve for PG-HER:

![alt text](https://user-images.githubusercontent.com/46422351/58496974-9f9bd680-8183-11e9-929e-679b2fb3ef6b.png)



And some examples of trajectories using a trained agent:

![alt text](https://user-images.githubusercontent.com/46422351/55673893-b1b17500-58b6-11e9-9293-364000ef4751.png)
![alt text](https://user-images.githubusercontent.com/46422351/55673896-b8d88300-58b6-11e9-8ced-4fe95394bd9b.png)
![alt text](https://user-images.githubusercontent.com/46422351/55673898-c4c44500-58b6-11e9-8a27-ffadcc98eb73.png)
![alt text](https://user-images.githubusercontent.com/46422351/55673901-d0177080-58b6-11e9-94a6-744ca3c52a85.png)
![alt text](https://user-images.githubusercontent.com/46422351/55673904-d6a5e800-58b6-11e9-8c76-8573d8781633.png)
![alt text](https://user-images.githubusercontent.com/46422351/55673910-e0c7e680-58b6-11e9-9f5d-3cf488c36318.png)


Not so evident in the trajectories shown here, but I noticed the agent tends to exploit the fact that the edges of the map are free by construction, and often maneuvers along the edges even if it's not mandatory.
