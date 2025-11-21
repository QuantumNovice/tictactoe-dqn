import gym
import numpy as np
from gym import spaces


class TicTacToeEnv(gym.Env):
    def __init__(self):
        super(TicTacToeEnv, self).__init__()
        # 3x3 grid. 0=Empty, 1=X, -1=O
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.float32)
        # 9 possible positions to place a mark
        self.action_space = spaces.Discrete(9)
        self.reset()

    def reset(self):
        self.board = np.zeros(9, dtype=np.float32)
        self.current_player = 1  # 1 for X, -1 for O
        self.done = False
        return self.board

    def step(self, action):
        if self.done:
            return self.board, 0, True, {}

        # Check validity
        if self.board[action] != 0:
            # Invalid move: severe penalty and end game (forces agent to learn rules)
            return self.board, -10, True, {"result": "Invalid"}

        # Apply move
        self.board[action] = self.current_player

        # Check for win
        if self.check_win(self.current_player):
            return self.board, 10, True, {"result": "Win"}

        # Check for draw
        if np.all(self.board != 0):
            return self.board, 0, True, {"result": "Draw"}

        # Switch player
        self.current_player *= -1

        # Return state, small reward for valid move to encourage prolonging game if not winning
        return self.board, 0, False, {}

    def check_win(self, player):
        b = self.board.reshape(3, 3)
        # Rows, Cols
        for i in range(3):
            if np.all(b[i, :] == player) or np.all(b[:, i] == player):
                return True
        # Diagonals
        if np.diag(b).sum() == 3 * player or np.fliplr(b).diagonal().sum() == 3 * player:
            return True
        return False

    def get_canonical_obs(self):
        # Returns board from the perspective of the current player
        # Current player always sees themselves as '1' and enemy as '-1'
        return self.board * self.current_player

    def render(self, mode="human"):
        symbols = {0: ".", 1: "X", -1: "O"}
        b = self.board.reshape(3, 3)
        print("-------------")
        for row in b:
            print(f"| {symbols[row[0]]} | {symbols[row[1]]} | {symbols[row[2]]} |")
            print("-------------")
