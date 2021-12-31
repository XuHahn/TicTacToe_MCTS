# 基于MCTS的井字棋对弈

## 1、环境配置：
```Bash
pip install -r requirements.txt
```
## 2、文件描述：
    TicTacToe_MCTS:
        conf.py: 配置文件
        game.py: 游戏类以及状态类
        Google.py: google SL与RL网络
        main.py: 运行游戏
        mcts.py: 蒙特卡洛搜索树
        visualization.py: 可视化文件
## 3、运行：
    1、在main.py文件中，修改主函数中的RunGame()参数，e.g:RunGame(300),数值表示难度取值范围[100,800]
```Python
if __name__ == '__main__':
    RunGame(300)  # 难度系数 100 - 800区间 
```
    2、改变players = {0: AI1, 1: human}的顺序决定先手/后手
```python
    players = {0: AI1, 1: human}  # 先手/后手
```
    0｜  1｜  2
    --------------
    0｜
    1｜
    2｜
    输入落子点：