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
    ret, observation = cv2.threshold(observation, 100, 1, cv2.THRESH_BINARY)  # 阈值，极化值
    index = np.argwhere(observation[:75, :] == 1).flatten()
    for i in range(len(index)):
        features[i] = index[i]
    return features   # transpose高维转置


if __name__ == '__main__':
    state = cv2.imread(r"C:\Users\lixin\Pictures\Saved Pictures\state.jpg", cv2.IMREAD_GRAYSCALE)
    _, state = cv2.threshold(state, 127, 1, cv2.THRESH_BINARY)
    index = np.argwhere(state[:75, :] == 1).flatten()
    print(index)
    # extract(state)
