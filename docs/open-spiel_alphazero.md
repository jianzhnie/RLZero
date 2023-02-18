### 概述：

Python 和 C++ 实现在概念上非常相似，并且具有大致相同的组件：actors 使用MCTS通过self-play 生成数据，evaluator 使用神经网络，learner 根据游戏更新网络，evaluator玩对比标准 MCTS 来衡量进展。

### MCTS

蒙特卡洛树搜索 (MCTS) 是一种用于玩许多游戏的通用搜索算法，但在 ~2005 年首次发现玩围棋的成功。它构建了一个由随机 rollout 引导的树，并且通常使用 UCT 来引导探索/开发权衡。对于我们的用例，我们用价值网络替换随机rollout 。我们使用策略网络代替统一先验, 使用 PUCT 而不是 UCT。

[我们在C++](https://github.com/deepmind/open_spiel/blob/master/open_spiel/algorithms/mcts.h)和 [python](https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/algorithms/mcts.py)中实现了 MCTS 。

### MCTS Evaluator

上面的两个 MCTS 实现都有一个可配置的Evaluator，它返回给定节点的值和先验策略。对于标准 MCTS，该值由随机推出给出，并且先验策略是统一的。对于 AlphaZero，值和先验由神经网络 evaluation 给出。AlphaZero evaluator采用模型，因此可以在训练期间使用或与经过训练的checkpoint一起使用 [open_spiel/python/examples/mcts.py](https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/examples/mcts.py)。

Both MCTS implementations above have a configurable evaluator that returns the value and prior policy of a given node. For standard MCTS the value is given by random rollouts, and the prior policy is uniform. For AlphaZero the value and prior are given by a neural network evaluation. The AlphaZero evaluator takes a model, so can be used during training or with a trained checkpoint for play with [open_spiel/python/examples/mcts.py](https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/examples/mcts.py).

### Actors

主脚本启动一组 actor 进程 (Python) 或线程 (C++)。actor创建两个具有共享evaluator和模型的 MCTS 实例，并玩self-play游戏，通过队列将trajectories传递给learner。actor越多，它生成训练数据的速度就越快，前提是您有足够的计算能力来实际运行它们。您的硬件的 actor 太多将意味着单个游戏的完成时间更长，因此您的数据相对于最新的checkpoint/权重可能更过时。

The main script launches a set of actor processes (Python) or threads (C++). The actors create two MCTS instances with a shared evaluator and model, and play self-play games, passing the trajectories to the learner via a queue. The more actors the faster it can generate training data, assuming you have sufficient compute to actually run them. Too many actors for your hardware will mean longer for individual games to finish and therefore your data could be more out of date with respect to the up to date checkpoint/weights.

### Learner

Learner从actors那里提取trajectories并将它们存储在固定大小的 FIFO replay buffer中。一旦replay buffer有足够的新数据，它就会从replay buffer中进行更新步骤采样。然后它保存一个checkpoint并更新所有actors的模型。它还使用一些统计信息更新`learner.jsonl`文件。

### Evaluators

主脚本还启动一组evaluator程序进程/线程。他们不断地与标准 MCTS+Solver 进行游戏，以了解训练的进展情况。MCTS 的对手可以根据他们每次移动的模拟次数来衡量实力，所以更高的级别意味着更强但更慢的对手。

### 输出

运行算法时，必须指定一个目录，所有输出都放在那里。

由于算法的并行性质，将日志写入 stdout/stderr 并不是很有用，因此每个actor/Learner/evaluator都将自己的日志文件写入配置目录。

checkpoint在每个更新步骤之后写入，大部分覆盖最新的一个，`checkpoint--1`但每个`checkpoint_freq`都保存在 `checkpoint-<step>`.

配置文件写入`config.json`, 使实验更具可重复性。

Learner还将 [jsonlines](http://jsonlines.org/)格式的机器可读日志写入`learner.jsonl`，可以用分析库读取。

## 用法：

### Python

代码位于[open_spiel/python/algorithms/alpha_zero/](https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/algorithms/alpha_zero/)。

最简单的示例针对一组训练步骤训练 tic_tac_toe 代理：

```shell
python3 open_spiel/python/examples/tic_tac_toe_alpha_zero.py
```

或者，您可以使用更多选项训练任意游戏：

```shell
python3 open_spiel/python/examples/alpha_zero.py --game connect_four --nn_model mlp --actors 10
```

### 分析

[在open_spiel/python/algorithms/alpha_zero/analysis.py](https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/algorithms/alpha_zero/analysis.py)中有一个分析库 ，它从实验（python 或 C++）中读取`config.json`和读取，并绘制图表损失、价值准确性、evaluator结果、actors速度、游戏长度等。它应该`learner.jsonl`将其变成 colab 是合理的。
