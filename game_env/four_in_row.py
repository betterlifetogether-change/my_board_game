from game_env.base_env import BaseEnv, VirtualBaseEnv
import numpy as np

class FourInRowEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        self.len_x, self.len_y, self.len_z = 5, 5, 5
        self.state_shape = [self.len_x, self.len_y, self.len_z]
        self.actions = set([i for i in range(self.len_x * self.len_y)])  # 所有可能动作的集合
        self.num_actions = self.len_x * self.len_y
        # 使用5*5*5矩阵表示棋盘状态, 0表示空, 1表示黑子, 2表示白子
        self.init_state = np.zeros([self.len_x, self.len_y, self.len_z], dtype=int)
        self.state = np.copy(self.init_state)  # 当前状态
        self.cur_player = 1  # 当前落子, 1为黑方, 2为白方
        self.out_of_height = False  # 上一步落子是否超出高度

    def get_next_state(self, action: int):
        # 输入下棋的位置action, 输出下棋后的棋盘状态
        assert action in self.actions
        # 将action转化为横纵坐标
        x = action % self.len_x
        y = action // self.len_y
        z = 0
        while self.state[x, y, z] > 0 and z < self.len_z:
            z += 1
        if z < self.len_z:
            # 在限定高度内才能正常落子, 否则视为停一手
            self.state[x, y, z] = self.cur_player
            self.out_of_height = False
        else:
            # 超出高度
            self.out_of_height = True
        # 黑白方切换
        if self.cur_player == 1:
            self.cur_player = 2
        else:
            self.cur_player = 1
        return {"state": self.state, "cur_player": self.cur_player}

    def get_reward(self, s1, a, s2):
        # TODO: 奖励函数, 在棋盘状态s1下落子a, 转移到s2下获得的奖励值
        # 一般定义为: 大于0时黑方优势, 小于0时白方优势
        return 0

    def is_game_over(self):
        # TODO: 根据self.state判断是否游戏结束
        return False

    def restart(self):
        self.state = np.copy(self.init_state)
        self.cur_player = 1
        return {"state": self.state, "cur_player": self.cur_player}

    def display(self):
        # 打印当前棋盘状态
        for z in range(self.len_z):
            if np.max(self.state[:, :, z]) <= 0:
                break
            for x in range(self.len_x):
                for y in range(self.len_y):
                    s = self.state[x, y, z]
                    if s == 0:
                        opt = "-"
                    elif s == 1:
                        opt = "1"
                    else:
                        opt = "2"
                    print(opt, end=" ")
                print()
            print(f"|___第{z + 1}层___|")


# 虚拟环境, 用于ai推理
class VirtualFourInRowEnv(FourInRowEnv, VirtualBaseEnv):
    def __init__(self):
        super().__init__()

    def roll_back(self, action: int):
        # 悔棋
        if not self.out_of_height:
            # 将action转化为横纵坐标
            x = action % self.len_x
            y = action // self.len_y
            z = 0
            while self.state[x, y, z] > 0 and z < self.len_z:
                z += 1
            if z > 0:
                self.state[x, y, z-1] = 0
        if self.cur_player == 1:
            self.cur_player = 2
        else:
            self.cur_player = 1

    def get_virtual_reward(self, s1, a, s2):
        # 根据活二/活三/活四的数量给奖励
        return 0


def test_play():
    game = FourInRowEnv()
    for i in range(100):
        action = input(f"第{i}步{'黑' if game.cur_player == 1 else '白'}方下:")
        game.step(int(action))
        game.display()


if __name__ == '__main__':
    test_play()
