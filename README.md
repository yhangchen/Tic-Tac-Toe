# Tic-Tac-Toe (Q & Deep Q)
Playing Tic-Tac-Toe with Q learning and Deep Q learning.

Achieve average reward **0.96** against the random player and **0.0** against the optimal player.

### Classes in tic_env.py:
- Environment
  - **TictactoeEnv**: Classical Tic-tac-toe game for two players.
  - **QlearningEnv**: an environment for Q learning, support learning from expert and learning by self-practice.
  - **DQNlearningEnv**: an environment for Deep Q learning, support learning from expert and learning by self-practice.
  - Important class method:
    - `backprop`: update player's Q (or do nothing for **OptimalPlayer**)
    - `train`: main training phase.
    - `test`: test against the random player or the optimal player.
    - `set_testing`: enable test during training.
    - `set_decay_eps`: enable decaying $\epsilon$ during training. 
    - These classes contain self.losses, self.training_reward_list, etc to store the statistics during training. 
    - Q table or Q network and replay memory is stored in the environment rather than in the player.
- Player
  - **OptimalPlayer**: an epsilon-greedy optimal player in Tic-tac-toe.
  - **QPlayer**: an epsilon-greedy Q learning player in Tic-tac-toe.
  - **DQNPlayer**: an epsilon-greedy Deep Q learning player in Tic-tac-toe.
  - Important class method:
    - `act`: action (random, or optimal, or based on Q)
    - `backprop`: update Q table in the Q learning.
    - `learn`: learning the Q function in the Deep Q learning.
    - We add some empty class method to **OptimalPlayer** to unify the way of calling.
- Others:
  - **ReplayMemory**: replay memory for deep Q learning.
  - **DQNNet**: neural network in deep Q learning.

### Examples
- Q learning from experts.
```python
Turns = np.array(['X','O'])
player1 = QPlayer(epsilon=0.1, player=Turns[0])
player2 = OptimalPlayer(epsilon=0.5, player=Turns[1])
env = QlearningEnv(player1=player1, player2=player2)
env.set_testing()
env.train(20000)
```
- Q learning from self practice.
```python
Turns = np.array(['X','O'])
player1 = QPlayer(epsilon=0.2, player=Turns[0])
player2 = QPlayer(epsilon=0.2, player=Turns[1])
env = QlearningEnv(player1=player1, player2=player2)
env.set_testing()
env.train(20000)
```

- Deep Q learning from experts.
```python
Turns = np.array(['X','O'])
player1 = DQNPlayer(epsilon=0.01, player=Turns[0])
player2 = OptimalPlayer(epsilon=0.5, player=Turns[1])
env = DQNlearningEnv(player1=player1, player2=player2)
env.set_testing()
env.train(20000)
```
- Deep Q learning from self practice.
```python
Turns = np.array(['X','O'])
player1 = DQNPlayer(epsilon=0.2, player=Turns[0])
player2 = DQNPlayer(epsilon=0.2, player=Turns[1])
env = DQNlearningEnv(player1=player1, player2=player2)
env.set_testing()
env.train(20000)
```

### Assignment
Yihang Chen: weite tic_env.py: implement Q and Deep Q learning and experiments interface; experiment on Q learning.

Yifei Song: experiment on Deep Q learning.


