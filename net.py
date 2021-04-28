import torch
from torch import nn


class AtariNet(nn.Module):

    def __init__(self, num_actions):
        super(AtariNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.hidden = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512, bias=True),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Sequential(
            nn.Linear(512, num_actions, bias=True)
        )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Conv2d:
            m.weight.data.normal_(0.0, 0.02)
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # 特征平铺
        x = self.hidden(x)
        x = self.out(x)
        return x


class CnnDQN(nn.Module):
    def __init__(self, inputs_shape, num_actions):
        super(CnnDQN, self).__init__()

        self.inut_shape = inputs_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(inputs_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.features_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def features_size(self):
        return self.features(torch.zeros(1, *self.inut_shape)).view(1, -1).size(1)


class Net(nn.Module):

    def __init__(self, num_states, num_actions):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(num_states,64, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Linear(64, num_actions, bias=True)
        )
        # self.norm1 = nn.BatchNorm1d(128)
        # self.dropout1 = nn.Dropout(p=0.25)
        # self.flatten2 = nn.Linear(128, 10)

        for m in self.features.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)

    def forward(self, x):
        x = self.features(x)
        x = torch.sigmoid(x)
        return x
