import pytest
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tictactoe_env import TicTacToeEnv


@pytest.fixture
def env():
    return TicTacToeEnv()


def test_initial_state(env):
    """Test if the board initializes correctly."""
    state = env.reset()
    assert np.all(state == 0)
    assert len(state) == 9
    assert env.current_player == 1
    assert env.done is False


def test_step_valid_move(env):
    """Test if a valid move updates the board and switches player."""
    env.reset()
    state, reward, done, info = env.step(0)  # Player 1 moves to index 0

    assert state[0] == 1
    assert env.current_player == -1  # Should switch to player -1
    assert reward == 0  # No immediate reward for non-winning move
    assert done is False


def test_invalid_move(env):
    """Test if placing a mark on an occupied spot is handled correctly."""
    env.reset()
    env.step(0)  # Occupy index 0

    # Try to move to index 0 again (current player is now -1)
    state, reward, done, info = env.step(0)

    assert reward == -10  # Penalty for invalid move
    assert done is True  # Game should end on invalid move (based on your env logic)
    assert info["result"] == "Invalid"


def test_win_row(env):
    """Test row win condition."""
    env.reset()
    # Manually set board for a row win for Player 1
    # X X X
    # . . .
    # . . .
    env.board = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)

    # Check win logic
    assert env.check_win(1) is True
    assert env.check_win(-1) is False


def test_win_diagonal(env):
    """Test diagonal win condition."""
    env.reset()
    # X . .
    # . X .
    # . . X
    env.board = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32)
    assert env.check_win(1) is True


def test_draw_condition(env):
    """Test if the game correctly identifies a draw."""
    env.reset()
    # X O X
    # X O X
    # O X O
    # Fill the board without a winner
    env.board = np.array([1, -1, 1, 1, -1, 1, -1, 1, -1], dtype=np.float32)

    # We need to trigger a step to check for draw
    # But since board is full, let's mock a step where board was almost full
    # Actually, simpler to call step on the last empty spot.

    # Let's construct a board with 1 empty spot that results in draw
    env.board = np.array([1, -1, 1, 1, -1, 1, -1, 1, 0], dtype=np.float32)
    env.current_player = -1

    state, reward, done, info = env.step(8)

    assert done is True
    assert info["result"] == "Draw"
    assert reward == 0
