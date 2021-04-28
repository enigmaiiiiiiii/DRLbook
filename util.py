import cv2
import numpy as np

def preprocess(observation):
    """
    image preprocess
    :param observation:
    :return:
    """
    observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
    observation = observation[26:110, :]  # 压缩图象，去掉比分栏
    ret, observation = cv2.threshold(observation, 65, 1, cv2.THRESH_BINARY)  # 阈值，极化值
    x = np.reshape(observation, (84, 84, 1))
    return x.transpose((2, 0, 1))  # transpose高维转置


def extract(observation):
    """
    image preprocess
    :param observation:
    :return:
    """
    features = np.zeros(120,dtype=int)
    observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
    observation = observation[26:110, :]  # 压缩图象，去掉比分栏
    ret, observation = cv2.threshold(observation, 65, 1, cv2.THRESH_BINARY)  # 阈值，极化值
    index = np.argwhere(observation[:75, :] == 1).flatten()
    if len(index) > 120:
        return index[:120].reshape((1,120))
    for i in range(len(index)):
        features[i] = index
    return features.reshape((1,120))   # transpose高维转置

def split3(observation):
    """
    image preprocess
    :param observation:
    :return:
    """
    try:
        features_1 = np.zeros(10,dtype=int)
        features_2 = np.zeros(10,dtype=int)
        features_3 = np.zeros(12,dtype = int)
        observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
        observation = observation[26:110, :]  # 压缩图象，去掉比分栏
        ret, observation = cv2.threshold(observation, 65, 1, cv2.THRESH_BINARY)  # 阈值，极化值
        index = np.argwhere(observation[:75, :] == 1)
        player1_feature = np.argwhere(observation[:75, 10] == 1)
        player2_feature = np.argwhere(observation[:75, 73] == 1)
        for x in [8,9,10,73,74,75]:
            index = np.delete(index,index[:,1] == x,axis = 0)
        ball_feature = index.flatten()
        features_1[:len(player1_feature)] = player1_feature
        features_2[:len(player1_feature)] = player2_feature
        features_3[:len(ball_feature)] = ball_feature
        return np.concatenate([features_1,features_2,features_3]).reshape((1,32))
    except :
        return np.zeros((1,32),dtype=int)

def extract_initial(observation):
    """
    image preprocess
    :param observation:
    :return:
    """
    features = np.zeros(120,dtype=int)
    observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
    observation = observation[26:110, :]  # 压缩图象，去掉比分栏
    ret, observation = cv2.threshold(observation, 100, 1, cv2.THRESH_BINARY)  # 阈值，极化值
    index = np.argwhere(observation[:75, :] == 1).flatten()
    for i in range(len(index)):
        features[i] = index[i]
    return features.reshape((1,120))   # transpose高维转置


def extract_tmp(observation):
    """
    image preprocess
    :param observation:
    :return:
    """

    observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
    observation = observation[26:110, :]  # 压缩图象，去掉比分栏
    ret, observation = cv2.threshold(observation, 65, 1, cv2.THRESH_BINARY)  # 阈值，极化值
    index = np.argwhere(observation[:75, :] == 1).flatten()

    return len(index)

if __name__ == '__main__':
    pass
    # extract(state)
