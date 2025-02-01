from game_env.base_env import BaseEnv
import numpy as np
from copy import deepcopy


class FourInRowEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        self.len_x, self.len_y, self.len_z = 5, 5, 5
        # 使用5*5*5矩阵表示棋盘状态, 0表示空, 1表示黑子, 2表示白子
        self.init_state = np.zeros([self.len_x, self.len_y, self.len_z], dtype=int)
        self.state = deepcopy(self.init_state)  # 当前状态
        self.cur_player = 1  # 当前落子, 1为黑方, 2为白方

    def get_next_state(self, action: int):
        # 输入下棋的位置action, 输出下棋后的棋盘状态
        assert 0 <= action < self.len_x * self.len_y
        # 将action转化为横纵坐标
        x = action % self.len_x
        y = action // self.len_y
        z = 0
        while self.state[x, y, z] > 0 and z < self.len_z:
            z += 1
        self.state[x, y, z] = self.cur_player
        # 黑白方切换
        if self.cur_player == 1:
            self.cur_player = 2
        else:
            self.cur_player = 1
        return self.state

    def get_reward(self, s1, a, s2):
        # TODO:    横
        a=b=c=h=0
        for x in range(5):
            y = 0
            z = 0
            while y < 5:
                var1 = self.state[x, y, z]
                if var1 == 1:
                    a = a + 1
                    if a > b:
                        b = a
                elif var1 == 2:
                    c = c + 1
                    if c > h:
                        h = c
                elif var1 == 0:
                    a = 0
                    c = 0
                if b >= 4:
                    print(f"黑棋胜利")
                    # 游戏结束
                if h >= 4:
                    print(f"白棋胜利")
                    # 游戏结束
                y = y + 1
        # TODO:    竖
        a=b=c=h=0
        for y in range(5):
            x = 0
            z = 0
            while x < 5:
                var2 = self.state[x, y, z]
                if var2 == 1:
                    a=a+1
                    if a > b:
                        b=a
                elif var2 == 2:
                    c=c+1
                    if c > h:
                        h=c
                elif var2 == 0:
                    a=0
                    c=0
                if b >= 4:
                    print(f"黑棋胜利")
                    #游戏结束
                if h >= 4:
                    print(f"白棋胜利")
                    #游戏结束
                x = x + 1
        # TODO:    TV对角1 俯视
        a = b = c = h = 0
        for x in range(5):
            z = 0
            while z < 5:
                i = 0
                while i < 10:
                    y = x + i - 5
                    if y < 0 or y >= 5:
                        continue
                    var3 = self.state[x, y, z]
                    if var3 == 1:
                        a = a + 1
                        if a > b:
                            b = a
                    elif var3 == 2:
                        c = c + 1
                        if c > h:
                            h = c
                    elif var3 == 0:
                        a = 0
                        c = 0
                    if b >= 4:
                        print(f"黑棋胜利")
                        # 游戏结束
                    if h >= 4:
                        print(f"白棋胜利")
                        # 游戏结束
                    i = i + 1
                z = z + 1
        # TODO:    TV对角2 俯视
        a = b = c = h = 0
        for x in range(5):
            z = 0
            while z < 5:
                i = 0
                while i < 10:
                    y = -x + i
                    if y < 0 or y >= 5:
                        continue
                    var4 = self.state[x, y, z]
                    if var4 == 1:
                        a = a + 1
                        if a > b:
                            b = a
                    elif var4 == 2:
                        c = c + 1
                        if c > h:
                            h = c
                    elif var4 == 0:
                        a = 0
                        c = 0
                    if b >= 4:
                        print(f"黑棋胜利")
                        # 游戏结束
                    if h >= 4:
                        print(f"白棋胜利")
                        # 游戏结束
                    i = i + 1
                z = z + 1
        # TODO:    FV对角1 主视X
        a = b = c = h = 0
        for x in range(5):
            y = 0
            while y < 5:
                i = 0
                while i < 10:
                    z = x + i - 5
                    if z < 0 or z >= 5:
                        continue
                    var3 = self.state[x, y, z]
                    if var3 == 1:
                        a = a + 1
                        if a > b:
                            b = a
                    elif var3 == 2:
                        c = c + 1
                        if c > h:
                            h = c
                    elif var3 == 0:
                        a = 0
                        c = 0
                    if b >= 4:
                        print(f"黑棋胜利")
                        # 游戏结束
                    if h >= 4:
                        print(f"白棋胜利")
                        # 游戏结束
                    i = i + 1
                y = y + 1
        # TODO:    FV对角2 主视X
        a = b = c = h = 0
        for x in range(5):
            y = 0
            while y < 5:
                i = 0
                while i < 10:
                    z = -x + i
                    if z < 0 or z >= 5:
                        continue
                    var4 = self.state[x, y, z]
                    if var4 == 1:
                        a = a + 1
                        if a > b:
                            b = a
                    elif var4 == 2:
                        c = c + 1
                        if c > h:
                            h = c
                    elif var4 == 0:
                        a = 0
                        c = 0
                    if b >= 4:
                        print(f"黑棋胜利")
                        # 游戏结束
                    if h >= 4:
                        print(f"白棋胜利")
                        # 游戏结束
                    i = i + 1
                y = y + 1
        # TODO:    SV对角1 侧视Y
        a = b = c = h = 0
        for z in range(5):
            x = 0
            while x < 5:
                i = 0
                while i < 10:
                    y = z + i - 5
                    if y < 0 or y >= 5:
                        continue
                    var3 = self.state[x, y, z]
                    if var3 == 1:
                        a = a + 1
                        if a > b:
                            b = a
                    elif var3 == 2:
                        c = c + 1
                        if c > h:
                            h = c
                    elif var3 == 0:
                        a = 0
                        c = 0
                    if b >= 4:
                        print(f"黑棋胜利")
                        # 游戏结束
                    if h >= 4:
                        print(f"白棋胜利")
                        # 游戏结束
                    i = i + 1
                x = x + 1
        # TODO:    SV对角2 侧视Y
        a = b = c = h = 0
        for z in range(5):
            x = 0
            while x < 5:
                i = 0
                while i < 10:
                    y = -z + i
                    if y < 0 or y >= 5:
                        continue
                    var4 = self.state[x, y, z]
                    if var4 == 1:
                        a = a + 1
                        if a > b:
                            b = a
                    elif var4 == 2:
                        c = c + 1
                        if c > h:
                            h = c
                    elif var4 == 0:
                        a = 0
                        c = 0
                    if b >= 4:
                        print(f"黑棋胜利")
                        # 游戏结束
                    if h >= 4:
                        print(f"白棋胜利")
                        # 游戏结束
                    i = i + 1
                x = x + 1
        # TODO:    高
        a = b = c = h = 0
        x = 0
        y = 0
        for z in range(5):
            while x < 5 and y < 5:
                var5 = self.state[x, y, z]
                if var5 == 1:
                    a = a + 1
                    if a > b:
                        b = a
                elif var5 == 2:
                    c = c + 1
                    if c > h:
                        h = c
                elif var5 == 0:
                    a = 0
                    c = 0
                if b >= 4:
                    print(f"黑棋胜利")
                    # 游戏结束
                if h >= 4:
                    print(f"白棋胜利")
                    # 游戏结束
                x = x + 1
                y = y + 1
        # TODO:     空间斜1
        a = b = c = h = 0
        i = 0
        while i < 5:
            j = 0
            while j < 5:
                k = 0
                while k < 5:
                    q = 0
                    while q < 5:
                        x = i + q
                        y = j + q
                        z = k + q
                        if x < 0 or x >= 5:
                            continue
                        if y < 0 or y >= 5:
                            continue
                        if z < 0 or z >= 5:
                            continue
                        var6 = self.state[x, y, z]
                        if var6 == 1:
                            a = a + 1
                            if a > b:
                                b = a
                        elif var6 == 2:
                            c = c + 1
                            if c > h:
                                h = c
                        elif var6 == 0:
                            a = 0
                            c = 0
                        if b >= 4:
                            print(f"黑棋胜利")
                            # 游戏结束
                        if h >= 4:
                            print(f"白棋胜利")
                            # 游戏结束
                        q = q + 1
                    k = k + 1
                j = j + 1
            i = i + 1
        # TODO:     空间斜2
        a = b = c = h = 0
        i = 0
        while i < 5:
            j = 0
            while j < 5:
                k = 0
                while k < 5:
                    q = 0
                    while q < 5:
                        x = i - q
                        y = j - q
                        z = k + q
                        if x < 0 or x >= 5:
                            continue
                        if y < 0 or y >= 5:
                            continue
                        if z < 0 or z >= 5:
                            continue
                        var6 = self.state[x, y, z]
                        if var6 == 1:
                            a = a + 1
                            if a > b:
                                b = a
                        elif var6 == 2:
                            c = c + 1
                            if c > h:
                                h = c
                        elif var6 == 0:
                            a = 0
                            c = 0
                        if b >= 4:
                            print(f"黑棋胜利")
                            # 游戏结束
                        if h >= 4:
                            print(f"白棋胜利")
                            # 游戏结束
                        q = q + 1
                    k = k + 1
                j = j + 1
            i = i + 1
        # TODO:     空间斜3
        a = b = c = h = 0
        i = 0
        while i < 5:
            j = 0
            while j < 5:
                k = 0
                while k < 5:
                    q = 0
                    while q < 5:
                        x = i - q
                        y = j + q
                        z = k + q
                        if x < 0 or x >= 5:
                            continue
                        if y < 0 or y >= 5:
                            continue
                        if z < 0 or z >= 5:
                            continue
                        var6 = self.state[x, y, z]
                        if var6 == 1:
                            a = a + 1
                            if a > b:
                                b = a
                        elif var6 == 2:
                            c = c + 1
                            if c > h:
                                h = c
                        elif var6 == 0:
                            a = 0
                            c = 0
                        if b >= 4:
                            print(f"黑棋胜利")
                            # 游戏结束
                        if h >= 4:
                            print(f"白棋胜利")
                            # 游戏结束
                        q = q + 1
                    k = k + 1
                j = j + 1
            i = i + 1
        # TODO:     空间斜4
        a = b = c = h = 0
        i = 0
        while i < 5:
            j = 0
            while j < 5:
                k = 0
                while k < 5:
                    q = 0
                    while q < 5:
                        x = i + q
                        y = j - q
                        z = k + q
                        if x < 0 or x >= 5:
                            continue
                        if y < 0 or y >= 5:
                            continue
                        if z < 0 or z >= 5:
                            continue
                        var6 = self.state[x, y, z]
                        if var6 == 1:
                            a = a + 1
                            if a > b:
                                b = a
                        elif var6 == 2:
                            c = c + 1
                            if c > h:
                                h = c
                        elif var6 == 0:
                            a = 0
                            c = 0
                        if b >= 4:
                            print(f"黑棋胜利")
                            # 游戏结束
                        if h >= 4:
                            print(f"白棋胜利")
                            # 游戏结束
                        q = q + 1
                    k = k + 1
                j = j + 1
            i = i + 1
        # TODO: 奖励函数, 在棋盘状态s1下落子a, 转移到s2下获得的奖励值
        # 一般定义为: 大于0时黑方优势, 小于0时白方优势
        return 0

    def is_game_over(self):
        # TODO: 根据self.state判断是否游戏结束
        return False

    def restart(self):
        # TODO: 恢复初始状态
        self.state = deepcopy(self.init_state)

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


def test_play():
    game = FourInRowEnv()
    for i in range(100):
        action = input(f"第{i}步{'黑' if game.cur_player == 1 else '白'}方下:")
        game.step(int(action))
        game.display()


if __name__ == '__main__':
    test_play()
