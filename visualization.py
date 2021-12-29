import seaborn as sns
import matplotlib.pyplot as plt
from main import run_game
from conf import *

data = {AI_1: 0, AI_2: 0, DRAW: 0}


def draw_fig(sim_1, sim_2, repeat=10000):
    for i in range(repeat):
        winner = run_game(sim_1, sim_2)
        data[winner] += 1
    sns.set_theme()
    result = list(data.values())
    fig = sns.barplot(x=['AI_1', 'AI_2', 'DRAW'], y=result)
    get_fig = fig.get_figure()
    get_fig.savefig(f'save_fig/rollout{sim_1}-{sim_2}|result{result[0]}-{result[1]}-{result[2]}', dpi=400)


if __name__ == '__main__':
    draw_fig(500, 100)
    draw_fig(200, 200)
