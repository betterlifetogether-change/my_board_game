import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
from copy import deepcopy
from ai.base_agent import BaseAgent


class DQN3D(nn.Module):
    def __init__(self, state_shape, num_actions, params, device):
        super(DQN3D, self).__init__()
        self.device = device
        self.state_shape = state_shape
        self.num_actions = num_actions

        # 3D 卷积层
        fig_dims = state_shape[-3:]  # 输入图像形状(长宽高, 不含通道)
        out_channels = params["net.out_channels"]
        kernel_sizes = params["net.kernel_sizes"]
        strides = params["net.strides"]
        paddings = params["net.paddings"]
        assert len(out_channels) == len(kernel_sizes) == len(strides) == len(paddings)
        self.use_bn = params["net.use_bn"]

        self.conv_layers = []
        self.bn_layers = []
        in_channels = 1
        for i in range(len(out_channels)):
            conv = nn.Conv3d(in_channels=in_channels,
                             out_channels=out_channels[i],
                             kernel_size=kernel_sizes[i],
                             stride=strides[i],
                             padding=paddings[i]).to(device)
            bn = nn.BatchNorm3d(out_channels[i]).to(device)
            in_channels = out_channels[i]  # 下一层的in为上一层的out
            fig_dims = [(fig_dim + 2 * paddings[i] - kernel_sizes[i]) // strides[i] + 1
                        for fig_dim in fig_dims]
            self.conv_layers.append(conv)
            if self.use_bn:
                self.bn_layers.append(bn)

        # 全连接层
        self.fc_layers = []
        self.dropout_layers = []
        last_hidden_dim = out_channels[-1]
        for d in fig_dims:
            last_hidden_dim *= d
        for i, hidden_dim in enumerate(params["net.fc_hidden_dims"]):
            self.fc_layers.append(nn.Linear(last_hidden_dim, hidden_dim).to(device))
            self.dropout_layers.append(nn.Dropout(params["net.dropout_p"]).to(device))
            last_hidden_dim = hidden_dim

        # 输出层
        self.fc_opt = nn.Linear(params["net.fc_hidden_dims"][-1], self.num_actions).to(device)

    def forward(self, x):
        # 3D 卷积层
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            if self.use_bn:
                x = self.bn_layers[i](x)

        # 展平
        x = x.view(x.size(0), -1)

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
        self.tar_q_net = deepcopy(self.q_net)
        assert params.get("train.buffer_size") is not None
        assert params.get("train.batch_size") is not None
        assert params.get("train.epsilon") is not None
        assert params.get("train.gamma") is not None
        assert params.get("train.lr") is not None
        assert params.get("train.tar_net_update_freq") is not None
        self.params = params
        self.replay_buffer = ReplayBuffer(params["train.buffer_size"], state_shape, device)
        self.num_actions = num_actions
        self.training_steps = 0
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=params["train.lr"])

    def set_train_mode(self):
        self.q_net.train()
        self.tar_q_net.train()

    def set_eval_mode(self):
        self.q_net.eval()
        self.tar_q_net.eval()

    def feature_fn(self, s, reverse=False):
        state = s["state"]
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(self.device)
        if state.dim() < 4:
            state = state.unsqueeze(0).unsqueeze(1)
        elif state.dim() == 4:
            state = state.unsqueeze(1)
        # 标准化, 白:-1,空:0, 黑:1
        replaced_state = torch.where(state == 2, -1, state)
        state = replaced_state
        # 黑白方视角转换
        cur_player = s["cur_player"]
        if reverse:
            cur_player = 1 if cur_player == 2 else 2
        if cur_player == 2:
            state = -state
        return state

    def get_action(self, s, training=False):
        if not training and random.random() < self.params["epsilon"]:
            a = random.choice(range(self.num_actions))
        else:
            s = self.feature_fn(s, reverse=True)
            with torch.no_grad():
                q_value = self.q_net(s)
            a = torch.argmax(q_value).cpu().item()
        return int(a)

    def learn(self, s1, a, r, s2, done):
        self.training_steps += 1
        s2["cur_player"] = s1["cur_player"]
        s1 = self.feature_fn(s1, reverse=True)
        s2 = self.feature_fn(s2, reverse=True)
        self.replay_buffer.add_data(s1, a, s2, r, done)
        if self.training_steps < self.replay_buffer.maxsize:
            # 数据未满时不训练
            return {}
        b_s1, b_a, b_s2, b_rs, b_done = self.replay_buffer.sample(self.params["train.batch_size"])
        ind = torch.LongTensor(range(b_a.shape[0]))
        Q = self.q_net(b_s1)[ind, b_a.squeeze(1)]
        gamma = params["train.gamma"]
        with torch.no_grad():
            Q_tar = torch.max(self.tar_q_net(b_s2), dim=1)[0]

        loss = torch.Tensor([0.0]).to(self.device)
        b_rs = b_rs.squeeze(1)
        b_done = b_done.squeeze(1)
        loss += 0.5 * nn.MSELoss()(Q, b_rs + gamma * Q_tar * (1 - b_done))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.training_steps % params["train.tar_net_update_freq"] == 0:
            self.update_target_network()
        return {"value_loss": loss.cpu().item()}

    def update_target_network(self):
        self.tar_q_net.load_state_dict(self.q_net.state_dict())

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
    agent.set_train_mode()
    for episode in range(params["train.num_episode"]):
        start_time = time.time()
        s1 = env.restart()
        loss_dict = {}
        episode_steps = 0
        episode_reward = 0
        for t in range(params["train.max_episode_step"]):
            episode_steps += 1
            a = agent.get_action(s1, True)
            s2, r, done = env.step(a)
            episode_reward += r
            loss_dict = agent.learn(s1, a, r, s2, done)
            if done:
                break
        if episode % params["train.test_per_episode"] == 0:
            # TODO: 和基线AI对比
            pass
        if episode % params["train.selfplay_update_per_episode"] == 0:
            # TODO: 自博弈模型更新
            pass
        if episode % 1000 == 0:
            print(f"Episode: {episode}, steps: {episode_steps}, reward: {episode_reward}, time: {time.time()-start_time}s")
            for k, v in loss_dict.items():
                print(f"{k}: {v}", end=", ")
            print()

if __name__ == '__main__':
    import os
    from utils.parse_conf import ParseConf

    config_path = os.path.join(os.path.dirname(__file__), "conf.yaml")
    params = ParseConf(config_path)
    start_train(params)
