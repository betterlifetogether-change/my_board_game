import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from copy import deepcopy
from ai.base_agent import BaseAgent


class DQN3D(nn.Module):
    def __init__(self, state_shape, num_actions, params, device):
        super(DQN3D, self).__init__()
        self.device = device
        self.state_shape = state_shape
        self.num_actions = num_actions

        # 3D 卷积层
        out_channels = params["net.out_channels"]
        kernel_size = params["net.kernel_size"]
        stride = params["net.stride"]
        padding = params["net.padding"]
        self.conv1 = nn.Conv3d(in_channels=1,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding).to(device)
        self.bn1 = nn.BatchNorm3d(out_channels).to(device)
        self.conv2 = nn.Conv3d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding).to(device)
        self.bn2 = nn.BatchNorm3d(out_channels).to(device)

        # 全连接层
        self.fc_layers = []
        self.dropout_layers = []
        last_hidden_dim = out_channels
        for d in self.state_shape:
            last_hidden_dim *= d
        for i, hidden_dim in enumerate(params["net.fc_hidden_dims"]):
            self.fc_layers.append(nn.Linear(last_hidden_dim, hidden_dim).to(device))
            self.dropout_layers.append(nn.Dropout(params["net.dropout_p"]).to(device))
            last_hidden_dim = hidden_dim

        # 输出层
        self.fc_opt = nn.Linear(params["net.fc_hidden_dims"][-1], self.num_actions).to(device)

    def forward(self, x):
        # 3D 卷积层
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # 展平
        x = x.view(x.size(0), -1)  # 展平成 (batch_size, 128 * length * height * width)

        # 全连接层
        for i in range(len(self.fc_layers)):
            x = F.relu(self.fc_layers[i](x))
            x = self.dropout_layers[i](x)

        # 输出层
        x = self.fc_opt(x)  # 输出形状: (batch_size, length * height * width)
        return x


class ReplayBuffer(object):
    def __init__(self, buffer_size, state_shape, device):
        maxsize = buffer_size
        self.maxsize = maxsize
        self.device = device
        self.S1 = torch.empty([maxsize] + state_shape, device=device)
        self.A = torch.empty([maxsize, 1], dtype=torch.long, device=device)
        self.S2 = torch.empty([maxsize] + state_shape, device=device)
        self.Rs = torch.empty([maxsize, 1], device=device)
        self.Done = torch.empty([maxsize, 1], dtype=torch.long, device=device)
        self.index = 0
        self.num_data = 0  # truly stored datas

    def add_data(self, s1, a, s2, reward, done):
        idx = self.index
        self.S1[idx] = torch.Tensor(s1)
        self.A[idx] = torch.LongTensor([a])
        self.S2[idx] = torch.Tensor(s2)
        self.Rs[idx] = torch.Tensor([reward])
        self.Done[idx] = torch.LongTensor([done])

        self.index = (self.index + 1) % self.maxsize
        self.num_data = min(self.num_data + 1, self.maxsize)

    def sample(self, batch_size):
        """Sample a batch of experiences."""
        device = self.device
        index = torch.randint(low=0, high=self.num_data, size=[batch_size], device=device)
        s1 = self.S1[index]
        a = self.A[index]
        s2 = self.S2[index]
        rs = self.Rs[index]
        done = self.Done[index]
        return s1, a, s2, rs, done


class DQNAgent(BaseAgent):
    def __init__(self, state_shape, num_actions, params):
        super().__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.q_net = DQN3D(state_shape, num_actions, params, device)
        self.tar_net = deepcopy(self.q_net)
        self.replay_buffer = ReplayBuffer(params["train.buffer_size"], [], device)
        self.params = params
        self.num_actions = num_actions

    def train(self):
        self.q_net.train()
        self.tar_net.train()

    def eval(self):
        self.q_net.eval()
        self.tar_net.eval()

    def feature_fn(self, s):
        state = s["state"]
        cur_player = s["cur_player"]
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(self.device)
        if state.dim() < 4:
            state = state.unsqueeze(0).unsqueeze(1)
        elif state.dim() == 4:
            state = state.unsqueeze(1)
        # 归一化, 白:-1,空:0, 黑:1
        if cur_player == 2:
            # 白方时黑白方置换, 永远是黑方视角
            replaced_state = torch.where(state == 1, -1, torch.where(state == 2, 1, state))
            state = replaced_state
        else:
            replaced_state = torch.where(state == 2, -1, state)
            state = replaced_state
        return state

    def get_action(self, s, training=False):
        if not training and random.random() < self.params["epsilon"]:
            a = random.choice(range(self.num_actions))
        else:
            s = self.feature_fn(s)
            with torch.no_grad():
                q_value = self.q_net(s)
            a = torch.argmax(q_value).cpu().item()
        return int(a)

    def learn(self, s1, a, r, s2, done):
        s1 = self.feature_fn(s1)
        s2 = self.feature_fn(s2)
        self.replay_buffer.add_data(s1, a, r, s2, done)



def start_train(params):
    import importlib
    # Get module and model class
    module_name = params["env.module_name"]
    class_name = params["env.class_name"]
    module = importlib.import_module(module_name)
    env_class = getattr(module, class_name)
    print(f"\n## from {module_name} import {class_name} Success.")
    env = env_class()
    print(f"## Build env({env}) Success.")

    agent = DQNAgent(env.state_shape, env.num_actions, params)
    agent.train()
    for episode in range(params["train.num_episode"]):
        s1 = env.restart()
        for t in range(params["train.max_episode_step"]):
            a = agent.get_action(s1, True)
            s2, r, done = env.step(a)
            agent.learn(s1, a, r, s2, done)
            if done:
                break
        if episode % params["train.test_per_episode"] == 0:
            # TODO: 和基线AI对比
            pass
        if episode % params["train.selfplay_update_per_episode"] == 0:
            # TODO: 自博弈模型更新
            pass


    
if __name__ == '__main__':
    import os
    from utils.parse_conf import ParseConf
    config_path = os.path.join(os.path.dirname(__file__), "conf.yaml")
    params = ParseConf(config_path)
    start_train(params)
