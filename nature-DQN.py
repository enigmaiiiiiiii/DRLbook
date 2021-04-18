import torch
import torch.nn as nn
from collections import deque
import numpy as np
import gym
import random
from net import AtariNet
from util import preprocess
import os
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import time
import pickle

writer = SummaryWriter('./losslog')

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent(object):
    def __init__(self, hotstart=True):
        self.network, self.target_network = AtariNet(ACTIONS_SIZE), AtariNet(ACTIONS_SIZE)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)  # 优化器只更新主网络参数，因为要从target网络中选择参数
        self.memory = deque()  # 数据量较多时，deque比list快？
        if hotstart and os.path.exists(".\\TrainedAgent\\state.pth") and os.path.exists(".\\memory\\nature_DQN_memory"):
            checkpoint = torch.load(".\\TrainedAgent\\state.pth")
            self.network.load_state_dict(checkpoint['network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.memory = pickle.load(open(".\\memory\\nature_DQN_memory", "wb"))  # 导入记忆
            # self.optimizer.load_state_dict(checkpoint['optimizer'])
            """只优化主网络参数"""
        self.learning_count = 0
        self.loss_func = nn.MSELoss()
        self.loss = torch.FloatTensor(0).to(device)
        self.network.to(device)
        self.target_network.to(device)

    def action(self, state, israndom):
        if israndom and random.random() < EPSILON:
            return np.random.randint(0, ACTIONS_SIZE)
        state = torch.unsqueeze(torch.FloatTensor(state), 0).cuda()
        actions_value = self.network.forward(state)  # 通过网络选择动作
        return torch.max(actions_value.cpu(), 1)[1].data.numpy()[0]

    def learn(self, state, action, reward, next_state, done):
        if done:
            self.memory.append((state, action, reward, next_state, 0))
        else:
            self.memory.append((state, action, reward, next_state, 1))
        if len(self.memory) > MEMORY_SIZE:
            self.memory.popleft()  # 删除队列最左侧元素，即最遥远的记忆
        if len(self.memory) < MEMORY_THRESHOLD:
            """每回合大约有1200个条记录加入memory,大约80个episode后开始训练神经网络model"""
            return
        elif len(self.memory) == MEMORY_THRESHOLD:
            print("第{}回合开始神经网路训练".format(i_episode))

        if self.learning_count % UPDATE_TIME == 0:
            self.target_network.load_state_dict(self.network.state_dict())
            """learning_count达到一定次数后
            通过load_state_dict函数target_network复制主网络network模型参数"""
        self.learning_count += 1

        batch = random.sample(self.memory, BATCH_SIZE)
        """每次从100000个回忆中取出BATCH_SIZE个样本进行训练"""


        state = torch.FloatTensor([x[0] for x in batch]).to(device)
        action = torch.LongTensor([[x[1]] for x in batch]).to(device)
        reward = torch.FloatTensor([[x[2]] for x in batch]).to(device)
        next_state = torch.FloatTensor([x[3] for x in batch]).to(device)
        done = torch.FloatTensor([[x[4]] for x in batch]).to(device)
        """tensor从cpu到GPU"""
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
        self.loss = self.loss_func(eval_q, target_q)

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def hotlearn(self, state, action, reward, next_state, done, i_episode):
        if done:
            self.memory.append((state, action, reward, next_state, 0))
        else:
            self.memory.append((state, action, reward, next_state, 1))
        if len(self.memory) > MEMORY_SIZE:
            self.memory.popleft()  # 删除队列最左侧元素，即最遥远的记忆
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

        start = time.time()  #
        batch = random.sample(self.memory, BATCH_SIZE)
        end = time.time()
        sample_time = end - start  #

        start = time.time()  #
        state = torch.FloatTensor([x[0] for x in batch]).to(device)
        action = torch.LongTensor([[x[1]] for x in batch]).to(device)
        reward = torch.FloatTensor([[x[2]] for x in batch]).to(device)
        next_state = torch.FloatTensor([x[3] for x in batch]).to(device)
        done = torch.FloatTensor([[x[4]] for x in batch]).to(device)
        """占用很长时间"""
        end = time.time()  #
        GPU_time = end - start

        """每次从100000个回忆中取出32个样本进行训练"""
        start = time.time()  #
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
        self.loss = self.loss_func(eval_q, target_q)

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        end = time.time()  #
        train_time = end - start
        # print("取样时间：{0:4f}\t转换时间：{1:4f}\t神经网络计算时间：{2:4f}".format(sample_time,GPU_time,train_time))
        if i_episode % 50 == 0:
            """每50局游戏保存一次参数"""
            train_state = {'network': self.network.state_dict(),
                           'target_network': self.target_network.state_dict(),
                           'optimizer': self.optimizer.state_dict(),
                           }
            torch.save(train_state, ".\\TrainedAgent\\state.pth")
            """每50回合固化记忆"""
            pickle.dump(self.memory, open(".\\memory\\nature_DQN_memory", "wb"))


agent = Agent()
imageno = 0
reward_num = 0
reward_time = time.time()

for i_episode in range(TOTAL_EPISODES):
    start_time = time.time()
    state = env.reset()
    state = preprocess(state)
    total = 0
    """像素状态处理，压缩转置"""
    while True:
        """一方分数达到20，done为True,while循环结束"""
        env.render()
        action = agent.action(state, True)
        next_state, reward, done, info = env.step(action)

        next_state = preprocess(next_state)
        start = time.time()
        agent.hotlearn(state, action, reward, next_state, done, i_episode)
        end = time.time()
        method_time = end-start
        total += method_time
        # if reward == 1:
        #     """
        #     对于Pong-v0这个游戏，state是像素，得分直接用作reward
        #     将图片数据裁剪，转置成模型输入数据tensor
        #     """
        #     reward_num += 1
        #     print("每得1分用时{}秒".format(time.time() - reward_time))  # 每次得分的耗时间隔
        #     writer.add_scalar("get_1_score_time_cost", time.time() - reward_time, reward_num)  # 将得分耗时间隔曲线记录在tensorboard
        #     reward_time = time.time()  # 得分时刻
        #
        #     state_img = Image.fromarray(state.squeeze())
        #     imageno += 1
        #     image_path = os.path.join(".\\image\\", str(imageno) + ".jpg")
        #     state_img.save(image_path, "jpeg")  # 将得分瞬间保存成图片
        #
        #     if agent.loss.size() != torch.Size([0]):
        #         writer.add_scalar("loss(reward=1)", agent.loss.cpu().detach().numpy(), reward_num)

        state = next_state
        if done:
            break
    time_1 = time.time() - start_time
    writer.add_scalar("time_1", time_1, i_episode)
    print("训练时间总时间：{0:.4f}\thotlearn方法总时间：{1:.4f}\t百分比：{2:.2f}%".format(time_1,total,total/time_1*100))
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

            # total_reward += reward

            state = next_state
            if done:
                break
        # print('episode: {} , total_reward: {}'.format(i_episode, round(total_reward, 3)))
env.close()
