from copy import deepcopy

import numpy as np
import pandas as pd

from game import *


class Node:
    def __init__(self, state: State, parent=None):
        self.state = deepcopy(state)
        self.parent = parent
        self.children = {}
        self.Q = 0
        self.N = 0

    def ucb(self, c=1.4):
        return self.Q / self.N + c * np.sqrt(np.log(self.N) / self.N) if self.N != 0 else np.Inf

    def expansion(self):
        for action in [tuple(i) for i in self.state.get_all_is_available_action()]:  # 枚举所有落子点
            cur_player = self.state.player  # 获取当前玩家
            next_board = self.state.to_numpy().copy()  # 获取当前棋盘上的情况
            next_board[action] = cur_player  # 枚举落子
            next_player = self.state.next_player(cur_player)  # 切换玩家
            next_state = State(next_board, next_player)
            child_node = Node(next_state, self)  # 将下一步state加入树中
            self.children[action] = child_node
        _, node = self.select(2)  # select一个节点返回，用于rollout
        return node

    def select(self, c_param=0):
        n = len(self.children.values())
        assert n > 0, '当前节点是叶节点'
        weights = [child.ucb(c_param) for child in self.children.values()]
        action = pd.Series(data=weights, index=self.children.keys()).idxmax()
        next_node = self.children[action]
        return action, next_node

    def backpropagation(self, winner, player):
        self.N += 1
        if winner == player:
            self.Q += 1
        elif winner == self.state.next_player(player):
            self.Q -= 1
        if self.parent is not None:
            self.parent.backpropagation(winner, player)

    def rollout(self):
        current_state = deepcopy(self.state)
        while True:
            is_win, winner = current_state.isWIn()
            if is_win:
                break
            action = current_state.random_action()
            current_state = current_state.take_action(action)
        return winner

    def is_full(self):
        return self.state.get_all_is_available_action().tolist() == [list(i) for i in self.children.keys()]


class MCTS:
    def __init__(self, player):
        self.root = None
        self.current_node = self.root
        self.player = player

    def simulation(self, repeat=1000):
        """
        :param repeat: rollout次数
        :return:
        """
        for _ in range(repeat):
            leaf_node = self.simulation_policy()  # 获取叶子节点
            winner = leaf_node.rollout()  # 对当前叶子节点进行rollout
            leaf_node.backpropagation(winner, self.player)  # 反向传播更新value

    def simulation_policy(self):
        current = self.current_node
        while True:
            is_win, winner = current.state.isWIn()
            if is_win:
                break
            if current.is_full():
                _, current = current.select(2)
            else:
                return current.expansion()
        return current

    def take_action(self, state: State, simulation_num=200):
        if not self.root:
            self.root = Node(state, None)
            self.current_node = self.root
        else:
            for child in self.current_node.children.values():
                if child.state == state:
                    self.current_node = child
                    break
            else:
                self.current_node = self.root
        self.simulation(simulation_num)
        action, next_node = self.current_node.select(0)
        self.current_node = next_node
        return action
