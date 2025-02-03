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
        self.max_height = 0  # 当前所有棋子的最大高度
        self.cur_height = 0  # 当前落子的高度
        self.game_over = False  # 是否游戏结束

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
            # 在限定高度内才能正常落子
            self.state[x, y, z] = self.cur_player
            self.out_of_height = False
            self.cur_height = z
            # 更新最大高度
            if z >= self.max_height:
                self.max_height = z
        else:
            # 超出高度直接判负
            self.out_of_height = True
            self.cur_height = self.len_z
            self.game_over = True
        # 黑白方切换
        if self.cur_player == 1:
            self.cur_player = 2
        else:
            self.cur_player = 1
        return {"state": self.state, "cur_player": self.cur_player}

    def get_max_len(self, sub_state: np.ndarray):
        cur_len1 = max_len1 = cur_len2 = max_len2 = 0
        for s in sub_state:
            if s == 1:
                cur_len1 += 1
                if cur_len1 > max_len1:
                    max_len1 = cur_len1
            elif s == 2:
                cur_len2 += 1
                if cur_len2 > max_len2:
                    max_len2 = cur_len2
            elif s ==0:
                cur_len1 = 0
                cur_len2 = 0
        return max_len1, max_len2

    def get_reward_one_row(self, sub_state: np.ndarray):
        len1, len2 = self.get_max_len(sub_state)
        if len1 >= 4:
            self.game_over = True
            return 1
        elif len2 >= 4:
            self.game_over = True
            return -1
        else:
            return 0

    def get_reward(self, s1, action, s2):
        if self.out_of_height:
            # 考虑到get_next_state已经转换了黑白方, 这里胜负结果取反
            return 1 if self.cur_player==1 else -1
        # 将action转化为横纵坐标
        cur_x = action % self.len_x
        cur_y = action // self.len_y
        cur_z = self.cur_height
        # 只需要针对当前坐标延伸的13个方向判定即可
        # x方向
        r = self.get_reward_one_row(self.state[:, cur_y, cur_z])
        if self.game_over: return r
        # y方向
        r = self.get_reward_one_row(self.state[cur_x, :, cur_z])
        if self.game_over: return r
        # z方向
        r = self.get_reward_one_row(self.state[cur_x, cur_y, :])
        if self.game_over: return r
        # (x,y)方向
        ts = [t for t in range(-3, 4)
              if 0 <= cur_x+t < self.len_x and 0 <= cur_y+t < self.len_y]
        r = self.get_reward_one_row(self.state[[cur_x+t for t in ts], [cur_y+t for t in ts], cur_z])
        if self.game_over: return r
        # TODO: 以下代码未完成
        # (x,-y)方向
        ts = [t for t in range(-3, 4)
              if 0 <= cur_x+t < self.len_x and 0 <= cur_y-t < self.len_y]
        r = self.get_reward_one_row(self.state[[cur_x+t for t in ts], [cur_y-t for t in ts], cur_z])
        if self.game_over: return r
        # (x,z)方向
        ts = [t for t in range(-3, 4)
              if 0 <= cur_x + t < self.len_x and 0 <= cur_z + t < self.len_z]
        r = self.get_reward_one_row(self.state[[cur_x+t for t in ts], cur_y, [cur_z+t for t in ts]])
        if self.game_over: return r
        # (x,-z)方向
        ts = [t for t in range(-3, 4)
              if 0 <= cur_x + t < self.len_x and 0 <= cur_z - t < self.len_z]
        r = self.get_reward_one_row(self.state[[cur_x+t for t in ts], cur_y, [cur_z-t for t in ts]])
        if self.game_over: return r
        # (y,z)方向
        ts = [t for t in range(-3, 4)
              if 0 <= cur_y + t < self.len_y and 0 <= cur_z + t < self.len_z]
        r = self.get_reward_one_row(self.state[cur_x, [cur_y+t for t in ts], [cur_z+t for t in ts]])
        if self.game_over: return r
        # (y,-z)方向
        ts = [t for t in range(-3, 4)
              if 0 <= cur_y + t < self.len_y and 0 <= cur_z - t < self.len_z]
        r = self.get_reward_one_row(self.state[cur_x, [cur_y+t for t in ts], [cur_z-t for t in ts]])
        if self.game_over: return r
        # (x,y,z)方向
        ts = [t for t in range(-3, 4)
              if 0 <= cur_x + t < self.len_x and 0 <= cur_y + t < self.len_y and 0 <= cur_z + t < self.len_z]
        r = self.get_reward_one_row(self.state[[cur_x+t for t in ts], [cur_y+t for t in ts], [cur_z+t for t in ts]])
        if self.game_over: return r
        # (x,y,-z)方向
        ts = [t for t in range(-3, 4)
              if 0 <= cur_x + t < self.len_x and 0 <= cur_y + t < self.len_y and 0 <= cur_z - t < self.len_z]
        r = self.get_reward_one_row(self.state[[cur_x+t for t in ts], [cur_y+t for t in ts], [cur_z-t for t in ts]])
        if self.game_over: return r
        # (x,-y,z)方向
        ts = [t for t in range(-3, 4)
              if 0 <= cur_x + t < self.len_x and 0 <= cur_y - t < self.len_y and 0 <= cur_z + t < self.len_z]
        r = self.get_reward_one_row(self.state[[cur_x+t for t in ts], [cur_y-t for t in ts], [cur_z+t for t in ts]])
        if self.game_over: return r
        # (x,-y,-z)方向
        ts = [t for t in range(-3, 4)
              if 0 <= cur_x + t < self.len_x and 0 <= cur_y - t < self.len_y and 0 <= cur_z - t < self.len_z]
        r = self.get_reward_one_row(self.state[[cur_x+t for t in ts], [cur_y-t for t in ts], [cur_z-t for t in ts]])
        if self.game_over: return r
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
        if game.game_over:
            print("游戏结束")

def test_play2():
    game = FourInRowEnv()
    actions = [18, 12, 12, 6, 6, 0, 6, 0, 0, 1, 0]
    for i in range(11):
        action = actions[i]
        game.step(int(action))
        game.display()
        if game.game_over:
            print("游戏结束")

if __name__ == '__main__':
    # test_play2()
    game = FourInRowEnv()
    game.state=np.array([[[5*y+x for z in range(5)] for y in range(5)] for x in range(5)])
    game.get_reward(None,0,None)
    # game.get_max_len(np.array([1,0,1,1,1]))