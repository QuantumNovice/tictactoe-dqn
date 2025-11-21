import json
import os
import time

GAME_LOG_DIR = "game_logs"

if not os.path.exists(GAME_LOG_DIR):
    os.makedirs(GAME_LOG_DIR)


def save_game_log(history, player1_type, player2_type, winner):
    """
    history: list of board states (lists or arrays)
    winner: 1 (X), -1 (O), or 0 (Draw)
    """
    timestamp = int(time.time())
    filename = f"{GAME_LOG_DIR}/game_{timestamp}.json"

    # Convert numpy arrays to lists for JSON serialization
    serializable_history = [s.tolist() if hasattr(s, "tolist") else s for s in history]

    data = {
        "timestamp": timestamp,
        "p1": player1_type,
        "p2": player2_type,
        "winner": winner,
        "moves": serializable_history,
    }

    with open(filename, "w") as f:
        json.dump(data, f)

    print(f"Game saved to {filename}")


def load_game_log(filepath):
    with open(filepath, "r") as f:
        return json.load(f)
