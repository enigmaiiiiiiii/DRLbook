import torch
import torch.nn as nn
from collections import deque
import numpy as np
import gym
import random
from net import Net
from util import *
import os
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import time
import pickle
from collections import namedtuple

Transition = namedtuple("Transition", ('state', 'action', 'reward', 'next_state', 'done'))
# writer = SummaryWriter('./losslog')

BATCH_SIZE = 32
LR = 0.001
START_EPSILON = 1.0
FINAL_EPSILON = 0.1
EPSILON = START_EPSILON
EXPLORE = 100000  # 探索指数越大遇到的情况越多
GAMMA = 0.99  # GAMMA越大，价值衰减越小，越关注长远价值
TOTAL_EPISODES = 10000000
MEMORY_SIZE = 100000  # 记忆容量
MEMORY_THRESHOLD = 50000  # 开始训练阈值
UPDATE_TIME = 10000
TEST_FREQUENCY = 1000
env = gym.make('Pong-v0')
env = env.unwrapped
ACTIONS_SIZE = env.action_space.n  # [0:5]  6
STATES_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent(object):
    def __init__(self, hotstart=True):
        self.network, self.target_network = Net(STATES_SIZE, ACTIONS_SIZE), Net(STATES_SIZE, ACTIONS_SIZE)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)  # 优化器只更新主网络参数，因为要从target网络中选择参数
        """显存占用,loss反向传播，更新参数网络权重Weight,优化器占用大量现存"""
        self.memory = deque(maxlen=MEMORY_SIZE)  # 数据量较多时，deque比list快？
        if hotstart and os.path.exists("D:\\PycharmProjects\\nature_DQN\\TrainedAgent\\state_1.pth"):
            checkpoint = torch.load("D:\\PycharmProjects\\nature_DQN\\TrainedAgent\\state_1.pth")
            self.network.load_state_dict(checkpoint['network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            # self.optimizer.load_state_dict(checkpoint['optimizer'])
            """只优化主网络参数"""
        if os.path.exists(".\\memory\\nature_DQN_memory"):
            self.memory = pickle.load(open(".\\memory\\nature_DQN_memory", "rb"))  # 导入记忆

        self.learning_count = 0
        self.loss_func = nn.MSELoss()
        self.loss = torch.tensor(0).to(device)
        self.network.to(device)
        self.target_network.to(device)

    def action(self, state, israndom):
        if israndom and random.random() < EPSILON:
            return np.random.randint(0, ACTIONS_SIZE)
        self.network.eval()
        actions_value = self.network.forward(state)  # 通过网络选择动作
        return torch.max(actions_value, 1)[1]

    def hotlearn(self, state, action, reward, next_state, done, i_episode):
        if done:
            self.memory.append([state, action, reward, next_state, torch.tensor([0]).to(device)])
        else:
            self.memory.append([state, action, reward, next_state, torch.tensor([1]).to(device)])
        if len(self.memory) < MEMORY_THRESHOLD:
            """每回合大约有1200个条记录加入memory,大约82个episode后开始训练神经网络model"""
            return
        elif len(self.memory) == MEMORY_THRESHOLD:
            print("第{}回合开始神经网络训练".format(i_episode))  # 输出神经网络训练开始回合

        if self.learning_count % UPDATE_TIME == 0:
            self.target_network.load_state_dict(self.network.state_dict())
            """learning_count达到一定次数后
            通过load_state_dict函数target_network复制主网络network模型参数"""
        self.learning_count += 1

        transitions = random.sample(self.memory, BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        """
        run时为什么变量会消失
        大量数据从cpu至gpu时会消耗大量时间"""
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).view(-1, 1)
        reward_batch = torch.cat(batch.reward).view(-1, 1)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.cat(batch.done).view(-1, 1)
        """占用很长时间"""
        """每次从100000个回忆中取出32个样本进行训练"""
        eval_q = self.network.forward(state_batch).gather(1, action_batch)  # 主网络Q
        """从主网络中返回动作价值"""
        next_q = self.target_network(next_state_batch).detach()
        """
        从target_network网络返回Q(s,a)
        当学习次数小于10000时，target_network没有得到过参数"""
        target_q = reward_batch + GAMMA * next_q.max(1)[0].view(BATCH_SIZE, 1) * done_batch
        """
        目标价值函数计算自target_network
        target_Q(s,a) = r + γ*maxQ(s,)"""
        self.loss = self.loss_func(eval_q, target_q)

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()


agent = Agent(hotstart=False)
imageno = 0

for i_episode in range(TOTAL_EPISODES):
    torch.cuda.empty_cache()
    start_time = time.time()
    env.reset()
    state = torch.FloatTensor(1,32).to(device)
    total = 0
    score = 0
    """像素状态处理，压缩转置"""
    while True:
        """一方分数达到21，done为True,while循环结束"""
        # env.render()
        action = agent.action(state, True)
        next_state, reward, done, info = env.step(action)
        score += reward
        action = torch.tensor([action]).to(device)
        next_state = split3(next_state)  # 训练特征

        next_state = torch.FloatTensor(next_state).to(device)
        reward = torch.tensor([reward]).to(device)

        start = time.time()
        agent.hotlearn(state, action, reward, next_state, done, i_episode)
        end = time.time()
        method_time = end - start
        total += method_time
        # if reward == 1:
        #     """
        #     对于Pong-v0这个游戏，state是像素，得分直接用作reward
        #     将图片数据裁剪，转置成模型输入数据tensor
        #     """
        #     reward_num += 1
        #     print("每得1分用时{}秒".format(time.time() - reward_time))  # 每次得分的耗时间隔

        state = next_state
        if done:
            break
    time_1 = time.time() - start_time
    # writer.add_scalar("time_1", time_1, i_episode)
    print("训练时间总时间：{0:.4f}\t"
          "hotlearn方法总时间：{1:.4f}\t"
          "百分比：{2:.2f}%\t"
          "i_episode:{3}\t"
          "loss:{4:.6f}\t"
          "得分：{5}".format(time_1,
                          total,
                          total / time_1 * 100,
                          i_episode,
                          agent.loss.item(),
                          score))
    if EPSILON > FINAL_EPSILON:
        EPSILON -= (START_EPSILON - FINAL_EPSILON) / EXPLORE
        if i_episode % 200 == 0 and i_episode != 0:
            """每50局游戏保存一次参数"""
            print("保存参数和记忆")
            train_state = {'network': agent.network.state_dict(),
                           'target_network': agent.target_network.state_dict(),
                           'optimizer': agent.optimizer.state_dict(),
                           }
            torch.save(train_state, "D:\\PycharmProjects\\nature_DQN\\TrainedAgent\\state_1.pth")
        #     """每200回合固化记忆"""
        #     pickle.dump(agent.memory, open(".\\memory\\nature_DQN_memory", "wb"))
    # TEST
    # if i_episode % TEST_FREQUENCY == 0:
    #     """每1000回合，do something"""
    #     state = env.reset()
    #     state = preprocess(state)
    #     state = torch.FloatTensor(state).to(device)
    #     total_reward = 0
    #     while True:
    #         # env.render()
    #         action = agent.action(state, israndom=False)
    #         next_state, reward, done, info = env.step(action)
    #         next_state = preprocess(next_state)
    #         next_state = torch.FloatTensor(next_state).to(device)
    #
    #         # total_reward += reward
    #
    #         state = next_state
    #         if done:
    #             break
    # print('episode: {} , total_reward: {}'.format(i_episode, round(total_reward, 3)))
env.close()
