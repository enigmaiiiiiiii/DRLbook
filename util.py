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
    ret, observation = cv2.threshold(observation, 100, 255, cv2.THRESH_BINARY) # 阈值，极化值
    x = np.reshape(observation, (84, 84, 1))
    return x.transpose((2, 0, 1))  # transpose高维转置
