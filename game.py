from conf import *
import numpy as np


class State:
    def __init__(self, board, player):
        self.__board = board
        self.player = player

    def __eq__(self, other):
        return True if (self.__board == other.__board).all() and self.player == other.player else False

    def get_all_is_available_action(self):
        """
        获得全部可落子坐标
        :return: [ [x1,y1], [x2,y2], ... ]
        """
        x = self.__board == 0
        return np.argwhere(x)

    def random_action(self):
        """
        随机选择一个落子位置
        :return: [x1,y1]
        """
        n = len(self.get_all_is_available_action())
        if n == 0:
            raise ValueError('棋盘已满！')
        action = self.get_all_is_available_action()
        choice = np.random.choice(n)
        return action[choice]

    def is_available(self, action: tuple):
        """
        判断该坐标是否合法
        :param action: （x1,y1)
        :return: True if available else False
        """
        x, y = action
        return True \
            if (self.__board[x, y] == 0) and (0 <= x <= 2) and (0 <= y <= 2) \
            else False

    @staticmethod
    def next_player(cur_player):
        return AI_1 if cur_player == AI_2 else AI_2

    def isWIn(self):
        """
        判断游戏是否结束
        :return: ture or false, winner
        """
        if (AI_1 * 3 in np.sum(self.__board, axis=1)) or \
                (AI_1 * 3 in np.sum(self.__board, axis=0)) or \
                (AI_1 * 3 == np.sum(self.__board.diagonal())) or \
                (AI_1 * 3 == np.sum(np.flip(self.__board, 1).diagonal())):
            return True, AI_1

        if (AI_2 * 3 in np.sum(self.__board, axis=1)) or \
                (AI_2 * 3 in np.sum(self.__board, axis=0)) or \
                (AI_2 * 3 == np.sum(self.__board.diagonal())) or \
                (AI_2 * 3 == np.sum(np.flip(self.__board, 1).diagonal())):
            return True, AI_2

        if (self.__board != 0).all():
            return True, DRAW

        return False, NOTWIN

    def take_action(self, action):
        x, y = action
        if self.is_available((x, y)):
            next_board = self.__board.copy()
            next_board[x, y] = self.player
            next_player = self.next_player(self.player)
            return State(next_board, next_player)
        else:
            raise ValueError('落子不合法！')

    def to_numpy(self):
        return self.__board

    def print_board(self):
        # os.system('cls' if 'nt' == os.name else 'clear')
        print('-------------------------')
        for i in self.__board:
            for j in i:
                print('-\t', CHESS_STYLE[j], end='\t')
            print('-', end='')
            print('\n-------------------------')


class Human:
    def __init__(self, player):
        self.player = player

    def take_action(self, state: State, simulation=0):
        while True:
            txt = input('- 输入落子点：')
            x, y = [int(_) for _ in txt.split(' ')]
            if state.is_available((x, y)):
                return [x, y]


class Game:
    def __init__(self, start_player=AI_1):
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
        player = start_player
        self.state = State(board, player)

    def take_action(self, action):
        self.state = self.state.take_action(action)
        return self.state
