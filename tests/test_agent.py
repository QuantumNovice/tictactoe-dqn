import pytest
import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dqn_agent import Agent


@pytest.fixture
def agent():
    return Agent(state_dim=9, action_dim=9)


def test_agent_initialization(agent):
    """Test if agent initializes networks correctly."""
    assert agent.state_dim == 9
    assert agent.action_dim == 9
    assert agent.epsilon == 1.0
    assert len(agent.memory) == 0


def test_agent_action_shape(agent):
    """Test if act returns a valid integer action."""
    state = np.zeros(9)
    action = agent.act(state)
    assert isinstance(action, int)
    assert 0 <= action < 9


def test_agent_epsilon_greedy(agent):
    """Test if epsilon works (exploration vs exploitation)."""
    state = np.zeros(9)

    # Force random (Exploration)
    agent.epsilon = 1.0
    # We can't deterministicly test random, but we run it to ensure no crash
    _ = agent.act(state)

    # Force greedy (Exploitation)
    agent.epsilon = 0.0
    action = agent.act(state)

    # With 0 input and uninitialized weights, it should be deterministic
    # We just ensure it returns an int
    assert isinstance(action, int)


def test_memory_buffer(agent):
    """Test if we can store experiences."""
    state = np.zeros(9)
    next_state = np.zeros(9)
    agent.remember(state, 0, 1, next_state, False)

    assert len(agent.memory) == 1
    assert agent.memory[0][1] == 0  # Action stored


def test_replay_learning(agent):
    """Test if replay function runs and returns a loss."""
    # Populate memory with dummy data to allow batch sampling
    state = np.zeros(9)
    next_state = np.zeros(9)

    # Agent needs at least batch_size to learn. Default is usually 32 or 64.
    # Let's force a small batch for testing or fill memory
    batch_size = 4
    for _ in range(10):
        agent.remember(state, np.random.randint(0, 9), 0, next_state, False)

    loss = agent.replay(batch_size)

    # It should return a loss value (float)
    assert isinstance(loss, float)
    assert loss >= 0


def test_model_save_load(agent, tmp_path):
    """Test saving and loading the model."""
    # tmp_path is a pytest fixture for a temporary directory
    save_path = tmp_path / "test_model.pth"

    # Save
    agent.save(str(save_path))
    assert save_path.exists()

    # Load
    success = agent.load(str(save_path))
    assert success is True
