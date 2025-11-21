from dqn_agent import Agent
from tictactoe_env import TicTacToeEnv
from utils import save_game_log

MODEL_PATH = "tictactoe_dqn.pth"


def get_human_action(state):
    valid = False
    while not valid:
        try:
            # Show board mapping 0-8
            print("0|1|2")
            print("3|4|5")
            print("6|7|8")
            choice = int(input("Enter move (0-8): "))
            if 0 <= choice <= 8 and state[choice] == 0:
                return choice
            else:
                print("Invalid move. Spot taken or out of bounds.")
        except ValueError:
            print("Please enter a number.")


def play_human_vs_human():
    env = TicTacToeEnv()
    state = env.reset()
    history = [state.copy()]
    done = False

    while not done:
        env.render()
        # Track who is playing THIS turn
        turn_player = env.current_player
        print(f"Player {'X' if turn_player == 1 else 'O'}'s turn")

        action = get_human_action(state)

        state, reward, done, info = env.step(action)
        history.append(state.copy())

        if done:
            env.render()
            if info["result"] == "Win":
                # The player who just moved (turn_player) is the winner
                winner = turn_player
                print(f"Player {'X' if winner == 1 else 'O'} Wins!")
            elif info["result"] == "Draw":
                winner = 0
                print("Draw!")
            else:
                winner = 0
                print("Game Over (Invalid Move)")

            save = input("Save replay? (y/n): ")
            if save.lower() == "y":
                save_game_log(history, "Human", "Human", winner)


def play_vs_ai(train_mode=True):
    env = TicTacToeEnv()
    agent = Agent(state_dim=9, action_dim=9)
    if not agent.load(MODEL_PATH):
        print("No pre-trained model found. AI will be random/untrained.")

    # User chooses side
    choice = input("Do you want to be X (1) or O (-1)? ")
    human_side = 1 if choice == "1" else -1

    state = env.reset()
    history = [state.copy()]
    done = False

    while not done:
        env.render()

        # Capture whose turn it is BEFORE stepping
        turn_player = env.current_player

        # Human Turn
        if turn_player == human_side:
            print("--- YOUR TURN ---")
            action = get_human_action(state)
        # AI Turn
        else:
            print("--- AI TURN ---")
            # AI sees board canonically (self = 1)
            canonical_state = state * turn_player
            # If training, use epsilon greedy, else pure max
            action = agent.act(canonical_state, eval_mode=not train_mode)

        # Execute
        next_state, reward, done, info = env.step(action)

        # If Learning Enabled and AI just played
        if train_mode and turn_player != human_side:
            # AI (turn_player) just made a move.
            # We reconstruct the experience for the AI.

            # State before move (canonical perspective of AI)
            c_state = state * turn_player
            # State after move (canonical perspective of AI)
            c_next_state = next_state * turn_player

            agent.remember(c_state, action, reward, c_next_state, done)
            loss = agent.replay(32)
            if loss:
                # Optional: Print less frequently to reduce clutter
                # print(f"[AI learned, loss: {loss:.4f}]")
                pass

        state = next_state
        history.append(state.copy())

        if done:
            env.render()
            res = info["result"]
            winner_val = 0

            if res == "Win":
                winner_val = turn_player  # The one who just moved won
                if winner_val == human_side:
                    print("You Win!")
                else:
                    print("AI Wins!")
            elif res == "Draw":
                print("Draw!")
            elif res == "Invalid":
                print("AI made invalid move. You win by default.")
                winner_val = human_side

            # Save Model if it learned
            if train_mode:
                agent.save(MODEL_PATH)
                print("AI Model Updated.")

            save = input("Save replay? (y/n): ")
            if save.lower() == "y":
                save_game_log(history, "Human", "AI", winner_val)


if __name__ == "__main__":
    print("1. Human vs Human")
    print("2. Human vs AI (Training Mode - AI learns from you)")
    print("3. Human vs AI (Eval Mode - AI plays its best)")

    m = input("Select Mode: ")
    if m == "1":
        play_human_vs_human()
    elif m == "2":
        play_vs_ai(train_mode=True)
    elif m == "3":
        play_vs_ai(train_mode=False)
