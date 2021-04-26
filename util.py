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
    player1_feature = index[index[:,1] == 10,0]
    player2_feature = index[index[:,1] == 73,0]
    if len(index) > 120:
        return index[:120].reshape((1,120))
    for i in range(len(index)):
        features[i] = index[i]
    return features.reshape((1,120))   # transpose高维转置

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
