import torch
import torch.nn as nn
from collections import deque
import numpy as np
import gym
import random
from net import AtariNet
from util import preprocess
import os

BATCH_SIZE = 32
LR = 0.001
START_EPSILON = 1.0
FINAL_EPSILON = 0.1
EPSILON = START_EPSILON
EXPLORE = 1000000
GAMMA = 0.99
TOTAL_EPISODES = 10000000
MEMORY_SIZE = 1000000
MEMORY_THRESHOLD = 100000
UPDATE_TIME = 10000
TEST_FREQUENCY = 1000
env = gym.make('Pong-v0')
env = env.unwrapped
ACTIONS_SIZE = env.action_space.n  # [0:5]  6

network = AtariNet(ACTIONS_SIZE)
target_network = AtariNet(ACTIONS_SIZE)


class Agent(object):
    def __init__(self, hotstart=True):
        self.network, self.target_network = AtariNet(ACTIONS_SIZE), AtariNet(ACTIONS_SIZE)
        if hotstart and os.path.exists(".\\TrainedAgent\\state.pth"):
            checkpoint = torch.load(".\\TrainedAgent\\state.pth")
            self.network.load_state_dict(checkpoint['network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.memory = checkpoint['memory']
        else:
            self.memory = deque()  # 数据量较多时，deque比list快？
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)
            """只优化主网络参数"""
        self.learning_count = 0
        self.loss_func = nn.MSELoss()

    def action(self, state, israndom):
        if israndom and random.random() < EPSILON:
            return np.random.randint(0, ACTIONS_SIZE)
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        actions_value = self.network.forward(state)  # 通过网络选择动作
        return torch.max(actions_value, 1)[1].data.numpy()[0]

    def learn(self, state, action, reward, next_state, done):
        if done:
            self.memory.append((state, action, reward, next_state, 0))
        else:
            self.memory.append((state, action, reward, next_state, 1))
        if len(self.memory) > MEMORY_SIZE:
            self.memory.popleft()  # 删除队列最左侧元素，即最遥远的记忆
        if len(self.memory) < MEMORY_THRESHOLD:
            """每回合大约有1200个条记录加入memory,大约90个episode后开始训练神经网络model"""
            return

        if self.learning_count % UPDATE_TIME == 0:
            self.target_network.load_state_dict(self.network.state_dict())
            """learning_count达到一定次数后
            通过load_state_dict函数target_network复制主网络network模型参数"""
        self.learning_count += 1

        batch = random.sample(self.memory, BATCH_SIZE)
        state = torch.FloatTensor([x[0] for x in batch])
        action = torch.LongTensor([[x[1]] for x in batch])
        reward = torch.FloatTensor([[x[2]] for x in batch])
        next_state = torch.FloatTensor([x[3] for x in batch])
        done = torch.FloatTensor([[x[4]] for x in batch])
        """每次从100000个回忆中取出32个样本进行训练"""
        eval_q = self.network.forward(state).gather(1, action)  # 主网络Q
        """从主网络中返回动作价值"""
        next_q = self.target_network(next_state).detach()
        """
        从target_network网络返回Q(s,a)
        当学习次数小于10000时，target_network没有得到过参数"""
        target_q = reward + GAMMA * next_q.max(1)[0].view(BATCH_SIZE, 1) * done
        """
        目标价值函数计算自target_network
        target_Q(s,a) = r + γ*maxQ(s,)"""
        loss = self.loss_func(eval_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def hotlearn(self, state, action, reward, next_state, done, i_episode):
        if done:
            self.memory.append((state, action, reward, next_state, 0))
        else:
            self.memory.append((state, action, reward, next_state, 1))
        if len(self.memory) > MEMORY_SIZE:
            self.memory.popleft()  # 删除队列最左侧元素，即最遥远的记忆
        if len(self.memory) < MEMORY_THRESHOLD:
            """每回合大约有1200个条记录加入memory,大约90个episode后开始训练神经网络model"""
            return

        if self.learning_count % UPDATE_TIME == 0:
            self.target_network.load_state_dict(self.network.state_dict())
            """learning_count达到一定次数后
            通过load_state_dict函数target_network复制主网络network模型参数"""
        self.learning_count += 1

        batch = random.sample(self.memory, BATCH_SIZE)
        state = torch.FloatTensor([x[0] for x in batch])
        action = torch.LongTensor([[x[1]] for x in batch])
        reward = torch.FloatTensor([[x[2]] for x in batch])
        next_state = torch.FloatTensor([x[3] for x in batch])
        done = torch.FloatTensor([[x[4]] for x in batch])
        """每次从100000个回忆中取出32个样本进行训练"""
        eval_q = self.network.forward(state).gather(1, action)  # 主网络Q
        """从主网络中返回动作价值"""
        next_q = self.target_network(next_state).detach()
        """
        从target_network网络返回Q(s,a)
        当学习次数小于10000时，target_network没有得到过参数"""
        target_q = reward + GAMMA * next_q.max(1)[0].view(BATCH_SIZE, 1) * done
        """
        目标价值函数计算自target_network
        target_Q(s,a) = r + γ*maxQ(s,)"""
        loss = self.loss_func(eval_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if i_episode % 100 == 0:
            """每100局游戏保存一次参数"""
            state = {'network': self.network.state_dict(),
                     'target_network': self.target_network.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'memory': self.memory
                     }
            torch.save(state, ".\\TrainedAgent\\state.pth")


agent = Agent()

for i_episode in range(TOTAL_EPISODES):
    state = env.reset()
    state = preprocess(state)
    """像素状态处理，压缩转置"""
    while True:
        env.render()
        action = agent.action(state, True)
        next_state, reward, done, info = env.step(action)
        next_state = preprocess(next_state)
        """
        对于Pong-v0这个游戏，state是像素，得分直接用作reward
        将图片数据裁剪，转置成模型输入数据tensor
        """
        agent.learn(state, action, reward, next_state, done)

        state = next_state
        if done:
            break
    if EPSILON > FINAL_EPSILON:
        EPSILON -= (START_EPSILON - FINAL_EPSILON) / EXPLORE

    # TEST
    if i_episode % TEST_FREQUENCY == 0:
        """每1000回合，do something"""
        state = env.reset()
        state = preprocess(state)
        total_reward = 0
        while True:
            # env.render()
            action = agent.action(state, israndom=False)
            next_state, reward, done, info = env.step(action)
            next_state = preprocess(next_state)

            total_reward += reward

            state = next_state
            if done:
                break
        print('episode: {} , total_reward: {}'.format(i_episode, round(total_reward, 3)))

env.close()
