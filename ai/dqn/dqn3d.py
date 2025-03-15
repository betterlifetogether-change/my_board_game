import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
from copy import deepcopy
from ai.base_agent import BaseAgent
from game_env.base_env import BaseEnv


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
        in_channels = state_shape[0]
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

        # 共享value层
        self.shared_value_layers = []
        last_hidden_dim = out_channels[-1]
        for d in fig_dims:
            last_hidden_dim *= d
        for i, hidden_dim in enumerate(params["net.fc_hidden_dims"]):
            self.shared_value_layers.append(nn.Linear(last_hidden_dim, hidden_dim).to(device))
            last_hidden_dim = hidden_dim
        self.shared_value_opt = nn.Linear(params["net.fc_hidden_dims"][-1], 1).to(device)

        # agent advantage头
        self.agent1_adv_layers = []
        last_hidden_dim = out_channels[-1]
        for d in fig_dims:
            last_hidden_dim *= d
        for i, hidden_dim in enumerate(params["net.fc_hidden_dims"]):
            self.agent1_adv_layers.append(nn.Linear(last_hidden_dim, hidden_dim).to(device))
            last_hidden_dim = hidden_dim
        self.agent1_adv_opt = nn.Linear(params["net.fc_hidden_dims"][-1], self.num_actions).to(device)

        self.agent2_adv_layers = []
        last_hidden_dim = out_channels[-1]
        for d in fig_dims:
            last_hidden_dim *= d
        for i, hidden_dim in enumerate(params["net.fc_hidden_dims"]):
            self.agent2_adv_layers.append(nn.Linear(last_hidden_dim, hidden_dim).to(device))
            last_hidden_dim = hidden_dim
        self.agent2_adv_opt = nn.Linear(params["net.fc_hidden_dims"][-1], self.num_actions).to(device)

    def forward(self, x):
        # 最后一个通道表示当前玩家, 1为agent1, 0为agent2
        cur_player = x[:, -1:, 0, 0, 0]
        # 3D 卷积层
        # TODO: 可以考虑跳跃连接
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            if self.use_bn:
                x = self.bn_layers[i](x)
            x = F.relu(x)

        flatten_fig = x.view(x.size(0), -1)

        v_x = flatten_fig
        for i in range(len(self.shared_value_layers)):
            v_x = F.relu(self.shared_value_layers[i](v_x))
        value = self.shared_value_opt(v_x)

        adv1_x = flatten_fig
        for i in range(len(self.agent1_adv_layers)):
            adv1_x = F.relu(self.agent1_adv_layers[i](adv1_x))
        adv1_x = self.agent1_adv_opt(adv1_x)
        adv1 = adv1_x - torch.mean(adv1_x, 1, keepdim=True)

        adv2_x = flatten_fig
        for i in range(len(self.agent2_adv_layers)):
            adv2_x = F.relu(self.agent2_adv_layers[i](adv2_x))
        adv2_x = self.agent1_adv_opt(adv2_x)
        adv2 = adv2_x - torch.mean(adv2_x, 1, keepdim=True)

        q_value = value + cur_player * adv1 + (1 - cur_player) * adv2

        return q_value


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
    def __init__(self, env_state_shape, num_actions, params, args):
        super().__init__()
        self.training = False
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # 特征工程: 将原始环境图像转为4通道
        state_shape = [4, ] + env_state_shape
        self.env_state_shape = env_state_shape
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
        self.is_warmup = args.warmup

        from game_env.four_in_row import VirtualFourInRowEnv
        self.v_env = VirtualFourInRowEnv()

    def set_train_mode(self):
        self.training = True
        self.q_net.train()
        self.tar_q_net.train()

    def set_eval_mode(self):
        self.training = False
        self.q_net.eval()
        self.tar_q_net.eval()

    def feature_fn(self, s):
        state = s["state"]
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(self.device)
        if state.dim() < 4:
            state = state.unsqueeze(0).unsqueeze(1)
        elif state.dim() == 4:
            state = state.unsqueeze(1)
        # 转化为4通道, 分别表示黑子, 白子, 所有子, 当前玩家
        black_state = torch.where(state == 1, 1.0, 0.0)
        white_state = torch.where(state == 2, 1.0, 0.0)
        chess_state = torch.where(state > 0, 1.0, 0.0)
        turn = 1 if s["cur_player"] == 1 else 0
        turn_state = turn * torch.ones_like(state, dtype=torch.float32)
        state = torch.concat([black_state, white_state, chess_state, turn_state], 1)

        return state

    def get_action(self, s):
        if self.training and self.training_steps < self.replay_buffer.maxsize:
            return random.choice(range(self.num_actions))
        if self.training and random.random() < self.params["train.epsilon"]:
            a = random.choice(range(self.num_actions))
        else:
            # 获取当前玩家下所有位置后的局势
            batch_x2, rs, dones, available_actions = self.get_s2_r_done_batch(s)

            with torch.no_grad():
                batch_q2 = self.q_net(batch_x2)
            max_q2 = torch.max(batch_q2, 1)[0]

            # max_q2是从对手角度考虑的价值,因此取argmin
            a_idx = torch.argmin((1-dones) * max_q2 + dones * rs).cpu().item()
            a = available_actions[a_idx]

        return int(a)

    def get_s2_r_done_batch(self, s):
        available_actions = [i for i in range(self.num_actions) if not s["forbidden_actions"][i]]
        x2s = []
        rs = []
        dones = []
        for action in available_actions:
            self.v_env.set_state(s)
            s2, r, done = self.v_env.step(action)
            x2 = self.feature_fn(s2)
            x2s.append(x2)
            rs.append(r)
            dones.append(done)
            self.v_env.roll_back(action)
        b_x2s = torch.concat(x2s, 0)
        b_rs = torch.tensor(rs, dtype=torch.float32, device=self.device)
        # 此时为ai推理对手,奖励取反
        if s["cur_player"] == 1:
            b_rs = -b_rs
        b_dones = torch.where(torch.tensor(dones, device=self.device), 1.0, 0.0)
        return b_x2s, b_rs, b_dones, available_actions

    def learn(self, s1, a, r, s2, done):
        self.training_steps += 1
        x1 = self.feature_fn(s1)
        x2 = self.feature_fn(s2)
        # 环境给的r为裁判视角(r<0白优),学习时从玩家视角(r>0为当前玩家优)
        if s1["cur_player"] == 2:
            r = -r
        self.replay_buffer.add_data(x1, a, x2, r, done)
        if self.training_steps < self.replay_buffer.maxsize and not self.is_warmup:
            # 数据未满时不训练
            return {}
        b_s1, b_a, b_s2, b_rs, b_done = self.replay_buffer.sample(self.params["train.batch_size"])
        ind = torch.LongTensor(range(b_a.shape[0]))
        Q = self.q_net(b_s1)[ind, b_a.squeeze(1)]
        gamma = self.params["train.gamma"]
        with torch.no_grad():
            Q_tar = torch.max(self.tar_q_net(b_s2), dim=1)[0]

        loss = torch.Tensor([0.0]).to(self.device)
        b_rs = b_rs.squeeze(1)
        b_done = b_done.squeeze(1)
        # 由于是minmax算法, 因此是r-maxQ
        loss += 0.5 * nn.MSELoss()(Q, b_rs - gamma * Q_tar * (1 - b_done))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.training_steps % self.params["train.tar_net_update_freq"] == 0:
            self.update_target_network()
        return {"value_loss": loss.cpu().item()}

    def update_target_network(self):
        self.tar_q_net.load_state_dict(self.q_net.state_dict())

    def load_checkpoint(self, ckpt):
        state_dict = torch.load(ckpt)
        self.q_net.load_state_dict(state_dict)
        self.tar_q_net.load_state_dict(state_dict)

    def save_checkpoint(self, ckpt):
        torch.save(self.q_net.state_dict(), ckpt)
