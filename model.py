import torch
import torch.nn as nn
from collections import deque
import numpy as np
import gym
import random
from net import AtariNet
from util import preprocess


class hot_model:
    def __init__(self, model_path):
        self.model_path = model_path

    def save(self, trained_model):
        torch.save(trained_model, self.model_path)

    def read(self, trained_model):
        torch.load(trained_model, self.model_path)
