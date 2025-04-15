
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

    def get_cur_state(self):
        chess = np.where(np.greater(self.state, 0), 1, 0)
        forbidden_state = np.greater_equal(np.sum(chess, 2), self.len_z)
        forbidden_actions = np.reshape(np.transpose(forbidden_state), [self.len_x * self.len_y])
        return {"state": np.copy(self.state), "cur_player": self.cur_player, "forbidden_actions": forbidden_actions}

    def get_next_state(self, action: int):
        # 输入下棋的位置action, 输出下棋后的棋盘状态
        assert action in self.actions
        # 将action转化为横纵坐标
        x = action % self.len_x
        y = action // self.len_y
        z = 0
        while z < self.len_z and self.state[x, y, z] > 0:
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
        return self.get_cur_state()

    def get_max_len(self, sub_state: np.ndarray):
        cur_len1 = max_len1 = cur_len2 = max_len2 = 0
        last_s = 0
        for s in sub_state:
            if s == 1:
                if s != last_s: cur_len1 = 0
                cur_len1 += 1
                if cur_len1 > max_len1:
                    max_len1 = cur_len1
            elif s == 2:
                if s != last_s: cur_len2 = 0
                cur_len2 += 1
                if cur_len2 > max_len2:
                    max_len2 = cur_len2
            else:
                cur_len1 = 0
                cur_len2 = 0
            last_s = s
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
        r = self.get_reward_one_row(s2[:, cur_y, cur_z])
        if self.game_over: return r
        # y方向
        r = self.get_reward_one_row(s2[cur_x, :, cur_z])
        if self.game_over: return r
        # z方向
        r = self.get_reward_one_row(s2[cur_x, cur_y, :])
        if self.game_over: return r
        # (x,y)方向
        ts = [t for t in range(-3, 4)
              if 0 <= cur_x+t < self.len_x and 0 <= cur_y+t < self.len_y]
        r = self.get_reward_one_row(s2[[cur_x+t for t in ts], [cur_y+t for t in ts], cur_z])
        if self.game_over: return r
        # (x,-y)方向
        ts = [t for t in range(-3, 4)
              if 0 <= cur_x+t < self.len_x and 0 <= cur_y-t < self.len_y]
        r = self.get_reward_one_row(s2[[cur_x+t for t in ts], [cur_y-t for t in ts], cur_z])
        if self.game_over: return r
        # (x,z)方向
        ts = [t for t in range(-3, 4)
              if 0 <= cur_x + t < self.len_x and 0 <= cur_z + t < self.len_z]
        r = self.get_reward_one_row(s2[[cur_x+t for t in ts], cur_y, [cur_z+t for t in ts]])
        if self.game_over: return r
        # (x,-z)方向
        ts = [t for t in range(-3, 4)
              if 0 <= cur_x + t < self.len_x and 0 <= cur_z - t < self.len_z]
        r = self.get_reward_one_row(s2[[cur_x+t for t in ts], cur_y, [cur_z-t for t in ts]])
        if self.game_over: return r
        # (y,z)方向
        ts = [t for t in range(-3, 4)
              if 0 <= cur_y + t < self.len_y and 0 <= cur_z + t < self.len_z]
        r = self.get_reward_one_row(s2[cur_x, [cur_y+t for t in ts], [cur_z+t for t in ts]])
        if self.game_over: return r
        # (y,-z)方向
        ts = [t for t in range(-3, 4)
              if 0 <= cur_y + t < self.len_y and 0 <= cur_z - t < self.len_z]
        r = self.get_reward_one_row(s2[cur_x, [cur_y+t for t in ts], [cur_z-t for t in ts]])
        if self.game_over: return r
        # (x,y,z)方向
        ts = [t for t in range(-3, 4)
              if 0 <= cur_x + t < self.len_x and 0 <= cur_y + t < self.len_y and 0 <= cur_z + t < self.len_z]
        r = self.get_reward_one_row(s2[[cur_x+t for t in ts], [cur_y+t for t in ts], [cur_z+t for t in ts]])
        if self.game_over: return r
        # (x,y,-z)方向
        ts = [t for t in range(-3, 4)
              if 0 <= cur_x + t < self.len_x and 0 <= cur_y + t < self.len_y and 0 <= cur_z - t < self.len_z]
        r = self.get_reward_one_row(s2[[cur_x+t for t in ts], [cur_y+t for t in ts], [cur_z-t for t in ts]])
        if self.game_over: return r
        # (x,-y,z)方向
        ts = [t for t in range(-3, 4)
              if 0 <= cur_x + t < self.len_x and 0 <= cur_y - t < self.len_y and 0 <= cur_z + t < self.len_z]
        r = self.get_reward_one_row(s2[[cur_x+t for t in ts], [cur_y-t for t in ts], [cur_z+t for t in ts]])
        if self.game_over: return r
        # (x,-y,-z)方向
        ts = [t for t in range(-3, 4)
              if 0 <= cur_x + t < self.len_x and 0 <= cur_y - t < self.len_y and 0 <= cur_z - t < self.len_z]
        r = self.get_reward_one_row(s2[[cur_x+t for t in ts], [cur_y-t for t in ts], [cur_z-t for t in ts]])
        if self.game_over: return r
        return 0

    def is_game_over(self):
        return self.game_over

    def restart(self):
        self.state = np.copy(self.init_state)
        self.cur_player = 1
        self.game_over = False
        return self.get_cur_state()

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

    def set_state(self, s: dict):
        self.state = s["state"]
        self.cur_player = s["cur_player"]

    def roll_back(self, action: int):
        # 悔棋
        if not self.out_of_height:
            # 将action转化为横纵坐标
            x = action % self.len_x
            y = action // self.len_y
            z = 0
            while z < self.len_z and self.state[x, y, z]:
                z += 1
            if z > 0:
                self.state[x, y, z-1] = 0
        if self.cur_player == 1:
            self.cur_player = 2
        else:
            self.cur_player = 1
        self.game_over = False

    def get_virtual_reward(self, s1, a, s2):
        # 根据活二/活三/活四的数量给奖励
        def calculate_score(max_len1, max_len2):
            score = 0
            if max_len1 == 4 :
                score=1000
            elif max_len1 == 3 :
                score=50
            elif max_len1 == 2 :
                score=5
            if max_len2 == 4 :
                score=1000
            elif max_len2 == 3 :
                score=100
            elif max_len2 == 2 :
                score=5
            return score
        def get_max_len_1( sub_state: np.ndarray):
            cur_len1 = max_len1 = cur_len2 = max_len2 = 0
            for s in sub_state:
                current_1 = 0
                gaps_1 = 1
                if s==1:
                    current_1 += 1
                elif s==0 and gaps_1>0:
                    current_1 += 1
                    gaps_1 -= 1
                else:
                    current_1 = 0
                    gaps_1 = 1
                current_2 = 0
                gaps_2 = 1
                if s == 2:
                    current_2 += 1
                elif s == 0 and gaps_2 > 0:
                    current_2 += 1
                    gaps_2 -= 1
                else:
                    current_2 = 0
                    gaps_2 = 1
                cur_len1 = current_1
                cur_len2 = current_2
                if s == 1:
                    if cur_len1 > max_len1:
                        max_len1 = cur_len1
                elif s == 2:
                    if cur_len2 > max_len2:
                        max_len2 = cur_len2
            return max_len1, max_len2
        cur_x = a % self.len_x
        cur_y = a // self.len_y
        cur_z = self.cur_height
        total_score = 0
        # x方向
        max_len1, max_len2 = get_max_len_1(self.state[:, cur_y, cur_z])
        total_score += calculate_score(max_len1, max_len2)
        # y方向
        max_len1, max_len2 = get_max_len_1(self.state[cur_x, :, cur_z])
        total_score += calculate_score(max_len1, max_len2)
        # z方向
        max_len1, max_len2 = get_max_len_1(self.state[cur_x, cur_y, :])
        total_score += calculate_score(max_len1, max_len2)
        # (x,y)方向
        ts = [t for t in range(-3, 4)
              if 0 <= cur_x + t < self.len_x and 0 <= cur_y + t < self.len_y]
        max_len1, max_len2 = get_max_len_1(self.state[[cur_x + t for t in ts], [cur_y + t for t in ts], cur_z])
        total_score += calculate_score(max_len1, max_len2)
        # TODO: 以下代码未完成
        # (x,-y)方向
        ts = [t for t in range(-3, 4)
              if 0 <= cur_x + t < self.len_x and 0 <= cur_y - t < self.len_y]
        max_len1, max_len2 = get_max_len_1(self.state[[cur_x + t for t in ts], [cur_y - t for t in ts], cur_z])
        total_score += calculate_score(max_len1, max_len2)
        # (x,z)方向
        ts = [t for t in range(-3, 4)
              if 0 <= cur_x + t < self.len_x and 0 <= cur_z + t < self.len_z]
        max_len1, max_len2 = get_max_len_1(self.state[[cur_x + t for t in ts], cur_y, [cur_z + t for t in ts]])
        total_score += calculate_score(max_len1, max_len2)
        # (x,-z)方向
        ts = [t for t in range(-3, 4)
              if 0 <= cur_x + t < self.len_x and 0 <= cur_z - t < self.len_z]
        max_len1, max_len2 = get_max_len_1(self.state[[cur_x + t for t in ts], cur_y, [cur_z - t for t in ts]])
        total_score += calculate_score(max_len1, max_len2)
        # (y,z)方向
        ts = [t for t in range(-3, 4)
              if 0 <= cur_y + t < self.len_y and 0 <= cur_z + t < self.len_z]
        max_len1, max_len2 = get_max_len_1(self.state[cur_x, [cur_y + t for t in ts], [cur_z + t for t in ts]])
        total_score += calculate_score(max_len1, max_len2)
        # (y,-z)方向
        ts = [t for t in range(-3, 4)
              if 0 <= cur_y + t < self.len_y and 0 <= cur_z - t < self.len_z]
        max_len1, max_len2 = get_max_len_1(self.state[cur_x, [cur_y + t for t in ts], [cur_z - t for t in ts]])
        total_score += calculate_score(max_len1, max_len2)
        # (x,y,z)方向
        ts = [t for t in range(-3, 4)
              if 0 <= cur_x + t < self.len_x and 0 <= cur_y + t < self.len_y and 0 <= cur_z + t < self.len_z]
        max_len1, max_len2 = get_max_len_1(
            self.state[[cur_x + t for t in ts], [cur_y + t for t in ts], [cur_z + t for t in ts]])
        total_score += calculate_score(max_len1, max_len2)
        # (x,y,-z)方向
        ts = [t for t in range(-3, 4)
              if 0 <= cur_x + t < self.len_x and 0 <= cur_y + t < self.len_y and 0 <= cur_z - t < self.len_z]
        max_len1, max_len2 = get_max_len_1(
            self.state[[cur_x + t for t in ts], [cur_y + t for t in ts], [cur_z - t for t in ts]])
        total_score += calculate_score(max_len1, max_len2)
        # (x,-y,z)方向
        ts = [t for t in range(-3, 4)
              if 0 <= cur_x + t < self.len_x and 0 <= cur_y - t < self.len_y and 0 <= cur_z + t < self.len_z]
        max_len1, max_len2 = get_max_len_1(
            self.state[[cur_x + t for t in ts], [cur_y - t for t in ts], [cur_z + t for t in ts]])
        total_score += calculate_score(max_len1, max_len2)
        # (x,-y,-z)方向
        ts = [t for t in range(-3, 4)
              if 0 <= cur_x + t < self.len_x and 0 <= cur_y - t < self.len_y and 0 <= cur_z - t < self.len_z]
        max_len1, max_len2 = get_max_len_1(
            self.state[[cur_x + t for t in ts], [cur_y - t for t in ts], [cur_z - t for t in ts]])
        total_score += calculate_score(max_len1, max_len2)
        return total_score

def human_vs_human():
    game = FourInRowEnv()
    for i in range(100):
        action = input(f"第{i}步{'黑' if game.cur_player == 1 else '白'}方下:")
        game.step(int(action))
        game.display()
        if game.game_over:
            print("游戏结束")

def take_actions(actions):
    game = FourInRowEnv()
    for action in actions:
        game.step(int(action))
        game.display()
        if game.game_over:
            print("游戏结束")

def human_vs_ai():
    game = FourInRowEnv()
    vgame = VirtualFourInRowEnv()
    from ai.greedy_agent import GreedyAgent
    agent = GreedyAgent(vgame)
    info = game.restart()
    s1 = info["state"]
    for i in range(100):
        if game.cur_player == 1:
            action = input(f"第{i}步黑方下:")
        else:
            action = agent.get_action(s1)
            print("ai下",action)
        s2, reward, done = game.step(int(action))
        game.display()
        s1 = s2




if __name__ == '__main__':
    # human_vs_ai()
    # game = FourInRowEnv()
    # game.state=np.array([[[5*y+x for z in range(5)] for y in range(5)] for x in range(5)])
    # game.get_reward(None,0,None)
    # game.get_max_len(np.array([1,0,1,1,1]))
    take_actions([2, 7, 11, 7, 19, 7, 7, 7])