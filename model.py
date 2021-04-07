import torch
import os
import torch.nn as nn
from collections import deque
import numpy as np
import gym
import random
from net import AtariNet
from util import preprocess


class hot_model:
    def __init__(self, model_name):

        self.model_path = os.path.join('.\\TrainedModels',model_name)

    def save(self, trained_model):
        torch.save(trained_model, self.model_path)

    def read(self):
        torch.load(self.model_path)
