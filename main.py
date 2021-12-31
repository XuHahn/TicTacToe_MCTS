from mcts import *


def RunGame(sim):
    AI1 = MCTS(AI_1)
    human = Human(AI_2)
    players = {0: AI1, 1: human}  # 先手/后手
    g = Game(start_player=players[0].player)
    turn = 0
    while True:
        current_state = g.state
        action = players[turn].take_action(current_state, sim)
        g.take_action(action)
        g.state.print_board()
        is_win, winner = g.state.isWIn()

        if is_win:
            return winner

        turn = (turn + 1) % 2


def AIGame(ai1_sim_num, ai2_sim_num):
    g = Game(start_player=AI_1)
    AI1 = MCTS(AI_1)
    AI2 = MCTS(AI_2)
    players = {0: AI1, 1: AI2}

    turn = 0
    while True:
        current_state = g.state
        action = players[turn].take_action(current_state,
                                           ai1_sim_num if turn == 0 else ai2_sim_num)
        g.take_action(action)
        is_win, winner = g.state.isWIn()

        if is_win:
            return winner
        turn = (turn + 1) % 2


if __name__ == '__main__':
    RunGame(200)  # 难度系数 100 - 800区间
