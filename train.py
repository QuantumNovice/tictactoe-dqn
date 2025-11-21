
from dqn_agent import Agent
from tictactoe_env import TicTacToeEnv

MODEL_PATH = "tictactoe_dqn.pth"
BATCH_SIZE = 64
EPISODES = 10000
SYNC_TARGET_EPISODES = 100


def train():
    env = TicTacToeEnv()
    # Input 9, Output 9 (positions)
    agent = Agent(state_dim=9, action_dim=9)

    start_episode = 0

    # Resume if model exists
    if agent.load(MODEL_PATH):
        print("Resuming training...")

    # Stats
    win_cnt = 0
    loss_cnt = 0
    draw_cnt = 0
    total_loss = 0

    print(f"Starting training for {EPISODES} episodes...")

    for e in range(start_episode, EPISODES + 1):
        state = env.reset()
        done = False

        while not done:
            # 1. Get Canonical State (Player sees themselves as 1)
            current_player_val = env.current_player
            canonical_state = state * current_player_val

            # 2. Act
            action = agent.act(canonical_state)

            # 3. Step
            next_state, reward, done, info = env.step(action)

            # 4. Store experience
            # Important: The reward returned is for the *current* player.
            # The Next State must also be canonical for the *same* player to learn next Q
            canonical_next_state = next_state * current_player_val

            # If move was invalid, game ends, big penalty
            if info.get("result") == "Invalid":
                agent.remember(canonical_state, action, reward, canonical_next_state, done)

            # If win, this player gets reward, store it
            elif info.get("result") == "Win":
                agent.remember(canonical_state, action, reward, canonical_next_state, done)

            else:
                # Normal move or Draw
                agent.remember(canonical_state, action, reward, canonical_next_state, done)

            # 5. Train
            loss = agent.replay(BATCH_SIZE)
            total_loss += loss

            state = next_state

            if done:
                res = info.get("result")
                if res == "Win":
                    # env.current_player is the winner (because env DOES NOT flip on win)
                    if env.current_player == 1:
                        win_cnt += 1
                    else:
                        loss_cnt += 1
                elif res == "Invalid":
                    loss_cnt += 1  # Treat invalid as loss for stats purposes
                elif res == "Draw":
                    draw_cnt += 1

        if e % SYNC_TARGET_EPISODES == 0:
            agent.update_target_network()
            agent.save(MODEL_PATH)
            avg_loss = total_loss / max(1, SYNC_TARGET_EPISODES)
            print(
                f"Ep: {e} | Epsilon: {agent.epsilon:.3f} | Wins (X): {win_cnt} | Draws: {draw_cnt} | Wins (O)/Invalid: {loss_cnt} | Avg Loss: {avg_loss:.4f}"
            )
            # Reset stats
            win_cnt = 0
            loss_cnt = 0
            draw_cnt = 0
            total_loss = 0


if __name__ == "__main__":
    train()
