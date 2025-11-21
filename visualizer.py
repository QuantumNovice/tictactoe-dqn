import json
import os
import time

LOG_DIR = "game_logs"


def render_board(board_list):
    # Clear screen command based on OS
    os.system("cls" if os.name == "nt" else "clear")

    symbols = {0: ".", 1: "X", -1: "O"}
    # Board is flat list of 9
    print("\n\n")
    print(f" {symbols[board_list[0]]} | {symbols[board_list[1]]} | {symbols[board_list[2]]} ")
    print("-----------")
    print(f" {symbols[board_list[3]]} | {symbols[board_list[4]]} | {symbols[board_list[5]]} ")
    print("-----------")
    print(f" {symbols[board_list[6]]} | {symbols[board_list[7]]} | {symbols[board_list[8]]} ")
    print("\n\n")


def play_replay():
    files = [f for f in os.listdir(LOG_DIR) if f.endswith(".json")]
    if not files:
        print("No replays found in game_logs/")
        return

    print("Available Replays:")
    for i, f in enumerate(files):
        print(f"{i}: {f}")

    try:
        idx = int(input("Select file index: "))
        filepath = os.path.join(LOG_DIR, files[idx])

        with open(filepath, "r") as f:
            data = json.load(f)

        moves = data["moves"]
        print(f"Replaying: {data['p1']} vs {data['p2']}")
        print(f"Winner: {data['winner']}")
        time.sleep(2)

        for move in moves:
            render_board(move)
            time.sleep(1.0)  # 1 second per frame

        print("Replay Finished.")

    except (ValueError, IndexError):
        print("Invalid selection.")


if __name__ == "__main__":
    play_replay()
