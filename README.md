# Tic-Tac-Toe (Q & Deep Q)
Playing Tic-Tac-Toe with Q learning and Deep Q learning.

Achieve average reward **0.96** against the **random player** and **0.0** against the **optimal player**.

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
    - `set_decay_eps`: enable decaying $\varepsilon$ during training. 
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

### Algorithm Details
#### Q learning
The training cosists of even number of games, and the first player is switches after every game. In the environment, there are two players, which could be **OptimalPlayer** or **QPlayer**, and there is a Q table in the environment. After finishing each game in training, the Q player will update the Q table according to the episode of the most recent game, and the Q player will act according to the Q table. We use $\varepsilon$ to encourage exploration during training. 

#### Deep Q learning
The training cosists of even number of games, and the first player is switches after every game. In the environment, there are two players, which could be **OptimalPlayer** or **DQNPlayer**, and there are two networks, policy network and target network in the environment. The environment also contain a replay buffer to store transitions. The DQN player will act according to the policy network. We update the policy network after every game has finished. The implementation of policy learning and action based on Q network is standard. The key point is how the transitions is stored in the replay buffer.

For each game, we assume the whole episode is $\{s_1,a_1,r_1,\cdots, s_n, a_n, r_n, {\rm None}\}$. The `None` denotes the end of an episode, which indicates either $a_n$ is an unavailable move or $s_{n+1}$ is the end of the game. We store transitions  $\{s_{2i-1}, a_{2i-1}, s_{2i+1}, r_{2i-1}\}$ and $\{s_{2i}, a_{2i}, s_{2i+2}, r_{2i}\}$ in the buffer, where the next state is the state after the opponent's movement, which could be `None`.  Since $\{s_{2i-1}, a_{2i-1}, s_{2i+1}, r_{2i-1}\}$ is player X's move, $\{s_{2i}, a_{2i}, s_{2i+2}, r_{2i}\}$ is player O's move. When push them to the buffer, we need to specify the current viewpoint, i.e.
```
s[2i-1] = grid2state(grid, "X“)
s[2i] = grid2state(grid, "O“)
```
Another thing to pay attention to is the sparse reward problem. For each episode, $\{s_1,a_1,r_1,\cdots, s_n, a_n, r_n, {\rm None}\}$, only $r_n$ can be non-zero. We propagate the reward backward. Notice the even and odd positions of states are viewed by two players, we propose $r_i=r_n (-\alpha)^{n-i}$, where $\alpha$ is a discount factor we choose to be $\frac{1}{2}$. 


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
Yihang Chen: write tic_env.py: implement Q and Deep Q learning, and experimental interface; experiment on Q learning; write the report.

Yifei Song: experiment on Deep Q learning; write the report.


