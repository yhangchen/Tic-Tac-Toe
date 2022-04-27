# import gym
import numpy as np
import random
from copy import deepcopy

class TictactoeEnv:
    '''
    Description:
        Classical Tic-tac-toe game for two players who take turns marking the spaces in a three-by-three grid with X or O.
        The player who succeeds in placing three of their marks in a horizontal, vertical, or diagonal row is the winner.

        The game is played by two players: player 'X' and player 'O'. Player 'x' moves first.

        The grid is represented by a 3x3 numpy array, with value in {0, 1, -1}, with corresponding values:
            0 - place unmarked
            1 - place marked with X
            -1 - place marked with O

        The game environment will recieve movement from two players in turn and update the grid.

    self.step:
        recieve the movement of the player, update the grid

    The action space is [0-8], representing the 9 positions on the grid.

    The reward is 1 if you win the game, -1 if you lose, and 0 besides.
    '''

    def __init__(self):
        self.grid = np.zeros((3,3))
        self.end = False
        self.winner = None
        self.player2value = {'X': 1, 'O': -1}
        self.num_step = 0
        self.current_player = 'X' # By default, player 'X' goes first

    def check_valid(self, position):
        ''' Check whether the current action is valid or not
        '''
        if self.end:
            raise ValueError('This game has ended, please reset it!')
        if type(position) is int:
            position = (int(position / 3), position % 3)
        elif type(position) is not tuple:
            position = tuple(position)

        return False if self.grid[position] != 0 else True

    def step(self, position, print_grid=False):
        ''' Receive the movement from two players in turn and update the grid
        '''
        # check the position and value are valid or not
        # position should be a tuple like (0, 1) or int [0-8]
        if self.end:
            raise ValueError('This game has ended, please reset it!')
        if type(position) is int:
            position = (int(position / 3), position % 3)
        elif type(position) is not tuple:
            position = tuple(position)
        if self.grid[position] != 0:
            raise ValueError('There is already a chess on position {}.'.format(position))

        # place a chess on the position
        self.grid[position] = self.player2value[self.current_player]
        # update
        self.num_step += 1
        self.current_player = 'X' if self.num_step % 2 == 0 else  'O'
        # check whether the game ends or not
        self.checkEnd()

        if print_grid:
            self.render()

        return self.grid.copy(), self.end, self.winner

    def get_current_player(self):
        return self.current_player

    def checkEnd(self):
        # check rows and cols
        if np.any(np.sum(self.grid, axis=0) == 3) or np.any(np.sum(self.grid, axis=1) == 3):
            self.end = True
            self.winner = 'X'
        elif np.any(np.sum(self.grid, axis=0) == -3) or np.any(np.sum(self.grid, axis=1) == -3):
            self.end = True
            self.winner = 'O'
        # check diagnols
        elif self.grid[[0,1,2],[0,1,2]].sum() == 3 or self.grid[[0,1,2],[2,1,0]].sum() == 3:
            self.end = True
            self.winner = 'X'
        elif self.grid[[0,1,2],[0,1,2]].sum() == -3 or self.grid[[0,1,2],[2,1,0]].sum() == -3:
            self.end = True
            self.winner = 'O'
        # check if all the positions are filled
        elif (self.grid == 0).sum() == 0:
            self.end = True
            self.winner = None # no one wins
        else:
            self.end = False
            self.winner = None

    def reset(self):
        # reset the grid
        self.grid = np.zeros((3,3))
        self.end = False
        self.winner = None
        self.num_step = 0
        self.current_player = 'X'

        return self.grid.copy(), self.end, self.winner

    def observe(self):
        return self.grid.copy(), self.end, self.winner

    def reward(self, player='X'):
        if self.end:
            if self.winner is None:
                return 0
            else:
                return 1 if player == self.winner else -1
        else:
            return 0

    def render(self):
        # print current grid
        value2player = {0: '-', 1: 'X', -1: 'O'}
        for i in range(3):
            print('|', end='')
            for j in range(3):
                print(value2player[int(self.grid[i,j])], end=' ' if j<2 else '')
            print('|')
        print()

class OptimalPlayer:
    '''
    Description:
        A class to implement an epsilon-greedy optimal player in Tic-tac-toe.

    About optimial policy:
        There exists an optimial policy for game Tic-tac-toe. A player ('X' or 'O') can win or at least draw with optimial strategy.
        See the wikipedia page for details https://en.wikipedia.org/wiki/Tic-tac-toe
        In short, an optimal player choose the first available move from the following list:
            [Win, BlockWin, Fork, BlockFork, Center, Corner, Side]

    Parameters:
        epsilon: float, in [0, 1]. This is a value between 0-1 that indicates the
            probability of making a random action instead of the optimal action
            at any given time.

    '''
    def __init__(self, epsilon=0.2, player='X'):
        self.epsilon = epsilon
        self.player = player # 'x' or 'O'

    def set_player(self, player = 'X', j=-1):
        self.player = player
        if j != -1:
            self.player = 'X' if j % 2 == 0 else 'O'

    def empty(self, grid):
        '''return all empty positions'''
        avail = []
        for i in range(9):
            pos = (int(i/3), i % 3)
            if grid[pos] == 0:
                avail.append(pos)
        return avail

    def center(self, grid):
        '''
        Pick the center if its available,
        if it's the first step of the game, center or corner are all optimial.
        '''
        if np.abs(grid).sum() == 0:
            # first step of the game
            return [(1, 1)] + self.corner(grid)

        return [(1, 1)] if grid[1, 1] == 0 else []

    def corner(self, grid):
        ''' Pick empty corners to move '''
        corner = [(0, 0), (0, 2), (2, 0), (2, 2)]
        cn = []
        # First, pick opposite corner of opponent if it's available
        for i in range(4):
            if grid[corner[i]] == 0 and grid[corner[3 - i]] != 0:
                cn.append(corner[i])
        if cn != []:
            return cn
        else:
            for idx in corner:
                if grid[idx] == 0:
                    cn.append(idx)
            return cn

    def side(self, grid):
        ''' Pick empty sides to move'''
        rt = []
        for idx in [(0, 1), (1, 0), (1, 2), (2, 1)]:
            if grid[idx] == 0:
                rt.append(idx)
        return rt

    def win(self, grid, val=None):
        ''' Pick all positions that player will win after taking it'''
        if val is None:
            val = 1 if self.player == 'X' else -1

        towin = []
        # check all positions
        for pos in self.empty(grid):
            grid_ = np.copy(grid)
            grid_[pos] = val
            if self.checkWin(grid_, val):
                towin.append(pos)

        return towin

    def blockWin(self, grid):
        ''' Find the win positions of opponent and block it'''
        oppon_val = -1 if self.player == 'X' else 1
        return self.win(grid, oppon_val)

    def fork(self, grid, val=None):
        ''' Find a fork opportunity that the player will have two positions to win'''
        if val is None:
            val = 1 if self.player == 'X' else -1

        tofork = []
        # check all positions
        for pos in self.empty(grid):
            grid_ = np.copy(grid)
            grid_[pos] = val
            if self.checkFork(grid_, val):
                tofork.append(pos)

        return tofork

    def blockFork(self, grid):
        ''' Block the opponent's fork.
            If there is only one possible fork from opponent, block it.
            Otherwise, player should force opponent to block win by making two in a row or column
            Amomg all possible force win positions, choose positions in opponent's fork in prior
        '''
        oppon_val = -1 if self.player == 'X' else 1
        oppon_fork = self.fork(grid, oppon_val)
        if len(oppon_fork) <= 1:
            return oppon_fork

        # force the opponent to block win
        force_blockwin = []
        val = 1 if self.player == 'X' else -1
        for pos in self.empty(grid):
            grid_ = np.copy(grid)
            grid_[pos] = val
            if np.any(np.sum(grid_, axis=0) == val*2) or np.any(np.sum(grid_, axis=1) == val*2):
                force_blockwin.append(pos)
        force_blockwin_prior = []
        for pos in force_blockwin:
            if pos in oppon_fork:
                force_blockwin_prior.append(pos)

        return force_blockwin_prior if force_blockwin_prior != [] else force_blockwin

    def checkWin(self, grid, val=None):
        # check whether the player corresponding to the val will win
        if val is None:
            val = 1 if self.player == 'X' else -1
        target = 3 * val
        # check rows and cols
        if np.any(np.sum(grid, axis=0) == target) or np.any(np.sum(grid, axis=1) == target):
            return True
        # check diagnols
        elif grid[[0,1,2],[0,1,2]].sum() == target or grid[[0,1,2],[2,1,0]].sum() == target:
            return True
        else:
            return False

    def checkFork(self, grid, val=None):
        # check whether the player corresponding to the val will fork
        if val is None:
            val = 1 if self.player == 'X' else -1
        target = 2 * val
        # check rows and cols
        rows = (np.sum(grid, axis=0) == target).sum()
        cols = (np.sum(grid, axis=1) == target).sum()
        diags = (grid[[0,1,2],[0,1,2]].sum() == target) + (grid[[0,1,2],[2,1,0]].sum() == target)
        if (rows + cols + diags) >= 2:
            return True
        else:
            return False

    def randomMove(self, grid):
        """ Chose a random move from the available options. """
        avail = self.empty(grid)

        return avail[random.randint(0, len(avail)-1)]

    def act(self, grid, **kwargs):
        """
        Goes through a hierarchy of moves, making the best move that
        is currently available each time (with probabitity 1-self.epsilon).
        A touple is returned that represents (row, col).
        """
        # whether move in random or not
        if random.random() < self.epsilon:
            return self.randomMove(grid)

        ### optimial policies

        # Win
        win = self.win(grid)
        if win != []:
            return win[random.randint(0, len(win)-1)]
        # Block win
        block_win = self.blockWin(grid)
        if block_win != []:
            return block_win[random.randint(0, len(block_win)-1)]
        # Fork
        fork = self.fork(grid)
        if fork != []:
            return fork[random.randint(0, len(fork)-1)]
        # Block fork
        block_fork = self.blockFork(grid)
        if block_fork != []:
            return block_fork[random.randint(0, len(block_fork)-1)]
        # Center
        center = self.center(grid)
        if center != []:
            return center[random.randint(0, len(center)-1)]
        # Corner
        corner = self.corner(grid)
        if corner != []:
            return corner[random.randint(0, len(corner)-1)]
        # Side
        side = self.side(grid)
        if side != []:
            return side[random.randint(0, len(side)-1)]

        # random move
        return self.randomMove(grid)
    
    def toState(self, grid):
        "Convert a grid to a state in tuple format as key"
        return tuple([int(x) for x in np.ravel(grid)])

    # we add the following empty method to accomodate QPlayer

    def add(self, grid, move):
        pass

    def backprop(self, reward):
        pass

    def reset(self):
        pass

    def decay_eps(self, epoch, epoch_star=1):
        pass

    def train(self):
        pass

    def eval(self):
        pass


from collections import defaultdict
class QPlayer(OptimalPlayer):
    def __init__(self, epsilon=0.2, player='X', Q=defaultdict(lambda: defaultdict(int)), lr=0.05, decay=0.99):
        super(QPlayer, self).__init__(epsilon=epsilon, player=player)
        self.Q = Q  # Q function. 
        self.records = [] # history of (state, action) record
        self.lr = lr # learning rate
        self.decay = decay # decaying factor
        self.eps_min = 0.1
        self.eps_max = 0.8
        self.training = True

    def reset(self):
        self.records = []
    
    def train(self): # train mode
        self.training = True
    
    def eval(self): # eval/test mode
        self.training = False
    
    def add(self, grid, move):
        self.records.append((self.toState(grid), move))
    
    def backprop(self, reward):
        for state, move in reversed(self.records):
            self.Q[state][move] += self.lr * (self.decay * reward - self.Q[state][move])
            reward = max(self.Q[state].values())
    
    def decay_eps(self, epoch, epoch_star=1):
        self.epsilon = max(self.eps_min, self.eps_max*(1-epoch/epoch_star))

    def act(self, grid): # rewrite act
        # whether move in random or not
        if self.training and random.random() < self.epsilon:
            return self.randomMove(grid)
        if self.player=='X':
            assert sum(sum(grid)) <= 0
        else:
            assert sum(sum(grid)) > 0

        best_move = []
        best_move_value = -np.Inf
        moves = self.empty(grid) # possible moves
        current_state = self.toState(grid)
        for move in moves:
            Q_s_a = self.Q[current_state][move]
            if Q_s_a == best_move_value:
                best_move.append(move)
            elif Q_s_a > best_move_value:
                best_move = [move]
                best_move_value = Q_s_a
        best_move = random.choice(best_move)
        return best_move

class QlearningEnv(TictactoeEnv):
    def __init__(self, player1: OptimalPlayer, player2: OptimalPlayer):
        super(QlearningEnv, self).__init__()
        self.player1 = player1 # player X at the each game
        self.player2 = player2 # player O at the each game
        self.training_reward_list = defaultdict(list) # see record_reward(self)
        self.test_reward_list = defaultdict(list) # see record_reward(self)
        self.test_avg_reward = defaultdict(list)
        self.decay_eps = False # whether decay epsilon
        self.testing = False 
    
    def backprop(self):
        # backpropagate reward to update Q. 
        if self.winner == 'X':
            self.player1.backprop(1)
            self.player2.backprop(-1)
        elif self.winner == 'O':
            self.player1.backprop(-1)
            self.player2.backprop(1)
        else:
            self.player1.backprop(0)
            self.player2.backprop(0)

    def reset_all(self):
        # reset env by clearing game board and empty player's history
        # but do not modify player's Q function.
        self.reset()
        self.player1.reset()
        self.player2.reset()

    def set_decay_eps(self, epoch_star=1):
        # enable decaying epsilon
        self.decay_eps = True
        self.epoch_star = epoch_star # n* in the paper
    
    def set_testing(self, test_eps=0, test_per_epoch=250):
        # enable test every test_per_epoch
        self.testing = True
        self.test_per_epoch = test_per_epoch
        self.test_player = None
        self.test_eps = test_eps

    def record_reward(self, training=True):
        # we alwys record reward for player 'X' at self.training_reward_list['X']
        # and record for player 'O' at self.training_reward_list['O'].
        #  No matter which player we are training. 
        # Similarly for self.test_reward_list
        if training:
            reward_list = self.training_reward_list
        else:
            reward_list = self.test_reward_list
        if self.winner == 'X':
            reward_list['X'].append(1)
            reward_list['O'].append(-1)
        elif self.winner == 'O':
            reward_list['X'].append(-1)
            reward_list['O'].append(1)
        else:
            reward_list['X'].append(0)
            reward_list['O'].append(0)
    
    def get_reward(self, player=1, training=True):
        # player=1 if we want to get reward from the 1st player while input
        # player=2 if 2nd
        def connect_lst(list1, list2):
            result = [None]*(len(list1)+len(list2))
            result[::2] = list1
            result[1::2] = list2
            return result

        if training:
            reward_list = self.training_reward_list
        else:
            reward_list = self.test_reward_list

        if player==1:
            return connect_lst(reward_list['X'][0::2], reward_list['O'][1::2])
        elif player==2:
            return connect_lst(reward_list['O'][0::2], reward_list['X'][1::2])
        else:
            raise NotImplementedError

        
    
    def train(self, epochs=1000):
        for epoch in range(epochs):
            self.player1.player = 'X'
            self.player2.player = 'O'
            self.player1.train()
            self.player2.train()
            self.reset_all()
            if self.decay_eps: # decay eps every epoch if self.decay_eps
                self.player1.decay_eps(epoch, self.epoch_star)
                self.player2.decay_eps(epoch, self.epoch_star)
            while True:
                p1_action = self.player1.act(self.grid)
                self.player1.add(self.grid, p1_action)
                self.step(p1_action)
                self.checkEnd()
                if self.end: # end after first player's move
                    self.backprop()
                    self.record_reward()
                    self.reset_all()
                    break
                else:
                    p2_action = self.player2.act(self.grid)
                    self.player2.add(self.grid, p2_action)
                    self.step(p2_action)
                    self.checkEnd()
                    if self.end: # end after second player's move
                        self.backprop()
                        self.record_reward()
                        self.reset_all()
                        break
            if self.testing and (epoch+1) % self.test_per_epoch == 0:
                if epoch % 2==0:
                    self.test_player = deepcopy(self.player1)
                else:
                    self.test_player = deepcopy(self.player2)
                self.test_player.player = 'X' # we always set the test player as 'X'
                self.test_player.epsilon = self.test_eps
                self.test_player.reset()
                self.test_avg_reward['random'].append(self.test(self.test_player, 'random'))
                self.test_avg_reward['optimal'].append(self.test(self.test_player, 'optimal'))
            # switch the 1st player after every game
            self.player1, self.player2 = self.player2, self.player1
        self.player1.player = 'X'
        self.player2.player = 'O'



    def test(self, player: OptimalPlayer, target_player_type: str, epochs=500):
        self.test_reward_list = defaultdict(list)
        if target_player_type == 'random':
            target_player = OptimalPlayer(epsilon=1.0, player='O')
        elif target_player_type == 'optimal':
            target_player = OptimalPlayer(epsilon=0.0, player='O')
        else:
            raise NotImplementedError

        for epoch in range(epochs):
            if epoch % 2 == 0:
                player1 = player
                player2 = target_player
            else:
                player1 = target_player
                player2 = player
            player1.player = 'X'
            player2.player = 'O'
            player1.eval()
            player2.eval()
            self.reset_all()
            while True:
                p1_action = player1.act(self.grid)
                self.step(p1_action)
                self.checkEnd()
                if self.end: # end after player 1's move
                    self.record_reward(training=False)
                    self.reset_all()
                    break
                else:
                    p2_action = player2.act(self.grid)
                    self.step(p2_action)
                    self.checkEnd()
                    if self.end: # end after player 2's move
                        self.record_reward(training=False)
                        self.reset_all()
                        break
        return sum(self.get_reward(player=1, training=False))/epochs
