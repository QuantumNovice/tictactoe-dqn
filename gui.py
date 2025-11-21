import pygame
import numpy as np
import sys
from tictactoe_env import TicTacToeEnv
from dqn_agent import Agent

# --- CONSTANTS ---
WIDTH, HEIGHT = 1000, 700  # Increased height for buttons
BOARD_SIZE = 400
CELL_SIZE = BOARD_SIZE // 3
OFFSET_X = 50
OFFSET_Y = 100

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (220, 50, 50)
BLUE = (50, 50, 220)
GREEN = (50, 200, 50)
YELLOW = (220, 220, 50)
DARK_BG = (30, 30, 30)
BTN_COLOR = (70, 70, 70)
BTN_HOVER = (100, 100, 100)

MODEL_PATH = "tictactoe_dqn.pth"


class Button:
    def __init__(self, x, y, w, h, text, action_code):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.action_code = action_code
        self.is_hovered = False

    def draw(self, screen, font):
        color = BTN_HOVER if self.is_hovered else BTN_COLOR
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, WHITE, self.rect, 2)

        text_surf = font.render(self.text, True, WHITE)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def check_hover(self, pos):
        self.is_hovered = self.rect.collidepoint(pos)

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)


class GameApp:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Tic Tac Toe DQN Visualizer")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 20)
        self.big_font = pygame.font.SysFont("Arial", 48)

        # Initialize Game Components
        self.env = TicTacToeEnv()
        self.agent = Agent(state_dim=9, action_dim=9)
        if self.agent.load(MODEL_PATH):
            print("Loaded existing model.")

        # Player Roles: User requested Human=O (-1), AI=X (1)
        self.human_side = -1
        self.ai_side = 1

        self.mode = "PLAY"  # Options: PLAY, TRAIN_WATCH, TRAIN_TURBO

        # Stats
        self.total_episodes = 0
        self.wins_ai = 0  # X
        self.wins_human = 0  # O
        self.current_loss = 0

        # UI Buttons
        btn_y = HEIGHT - 80
        self.buttons = [
            Button(50, btn_y, 150, 50, "Restart Game", "RESET"),
            Button(220, btn_y, 150, 50, "Train (Watch)", "TRAIN_WATCH"),
            Button(390, btn_y, 150, 50, "Train (Turbo)", "TRAIN_TURBO"),
            Button(560, btn_y, 150, 50, "Stop / Play", "PLAY"),
            Button(730, btn_y, 150, 50, "Save Model", "SAVE"),
        ]

        self.reset_game()

    def reset_game(self):
        self.state = self.env.reset()
        self.done = False
        self.game_result = ""

        # If AI is X (1) and X goes first, AI acts immediately in PLAY mode
        if self.mode == "PLAY" and self.env.current_player == self.ai_side:
            # Slight delay for visual effect if desired, but here we just move
            self.step_ai_logic(eval_mode=True)

    def run(self):
        running = True
        while running:
            self.screen.fill(DARK_BG)
            mouse_pos = pygame.mouse.get_pos()

            # Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEMOTION:
                    for btn in self.buttons:
                        btn.check_hover(mouse_pos)

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Check Buttons
                    for btn in self.buttons:
                        if btn.is_clicked(mouse_pos):
                            self.handle_button(btn.action_code)

                    # Check Board Click (Only if Human Turn and Playing)
                    if self.mode == "PLAY" and not self.done:
                        if self.env.current_player == self.human_side:
                            if (
                                OFFSET_X < mouse_pos[0] < OFFSET_X + BOARD_SIZE
                                and OFFSET_Y < mouse_pos[1] < OFFSET_Y + BOARD_SIZE
                            ):
                                col = (mouse_pos[0] - OFFSET_X) // CELL_SIZE
                                row = (mouse_pos[1] - OFFSET_Y) // CELL_SIZE
                                idx = row * 3 + col
                                self.step_human(idx)

            # --- LOGIC LOOPS ---

            if self.mode == "TRAIN_WATCH":
                # Train 1 step per frame
                if self.done:
                    self.reset_and_track_train()
                else:
                    self.step_ai_logic(eval_mode=False)  # Self Play

            elif self.mode == "TRAIN_TURBO":
                # Train 500 steps per frame (Visualizes snapshot)
                steps = 0
                while steps < 500:
                    if self.done:
                        self.reset_and_track_train()
                    self.step_ai_logic(eval_mode=False)
                    steps += 1

            elif self.mode == "PLAY":
                # AI Turn in Play Mode
                if not self.done and self.env.current_player == self.ai_side:
                    # Add a small artificial delay for "thinking" feel only if needed
                    # But for responsiveness, we usually just go.
                    self.step_ai_logic(eval_mode=True)

            # --- DRAWING ---
            self.draw_board()
            self.draw_pieces()
            self.draw_dqn_viz()
            self.draw_info()

            for btn in self.buttons:
                btn.draw(self.screen, self.font)

            pygame.display.flip()
            self.clock.tick(60)  # Keep UI responsive

        pygame.quit()
        sys.exit()

    def handle_button(self, code):
        if code == "RESET":
            self.reset_game()
            # Reset stats if manually resetting? maybe not total episodes
        elif code == "TRAIN_WATCH":
            self.mode = "TRAIN_WATCH"
            self.reset_game()
        elif code == "TRAIN_TURBO":
            self.mode = "TRAIN_TURBO"
            self.reset_game()
        elif code == "PLAY":
            self.mode = "PLAY"
            self.reset_game()
        elif code == "SAVE":
            self.agent.save(MODEL_PATH)
            print("Model Saved manually.")

    def reset_and_track_train(self):
        self.total_episodes += 1
        if self.total_episodes % 100 == 0:
            self.agent.update_target_network()
        self.state = self.env.reset()
        self.done = False

    def step_human(self, action_idx):
        if self.state[action_idx] != 0:
            return  # Invalid

        # Execute Human Move
        self.execute_move(action_idx)

    def step_ai_logic(self, eval_mode=True):
        # Get Current Player Perspective
        turn_player = self.env.current_player
        canonical_state = self.state * turn_player

        # Epsilon handling
        # If Training (WATCH or TURBO), we use agent's internal epsilon
        # If PLAY, we force eval_mode=True (Pure Greedy)
        action = self.agent.act(canonical_state, eval_mode=eval_mode)

        # Failsafe: If AI picks invalid move (rare but possible early in training), pick random valid
        if self.state[action] != 0:
            valid_indices = [i for i, x in enumerate(self.state) if x == 0]
            if valid_indices:
                action = np.random.choice(valid_indices)
            else:
                return  # Game is full/done

        self.execute_move(action)

    def execute_move(self, action_idx):
        # Capture Pre-Move Info for Learning
        turn_player = self.env.current_player
        canonical_state = self.state * turn_player

        # Env Step
        next_state, reward, done, info = self.env.step(action_idx)

        # Capture Post-Move Info (Canonical for the player who just moved)
        canonical_next_state = next_state * turn_player

        # Store & Learn (Only learn if we are in a training mode OR if Human wants AI to learn during play)
        # NOTE: The user prompt implies "AI learns from it" in play mode too.
        # So we ALWAYS learn.
        self.agent.remember(
            canonical_state, action_idx, reward, canonical_next_state, done
        )
        loss = self.agent.replay(64)
        if loss:
            self.current_loss = loss

        # Update Game State
        self.state = next_state
        self.done = done

        if done:
            res = info["result"]
            if res == "Win":
                winner = turn_player
                self.game_result = f"Winner: {'X (AI)' if winner == 1 else 'O (Human)'}"
                if winner == 1:
                    self.wins_ai += 1
                else:
                    self.wins_human += 1
            elif res == "Draw":
                self.game_result = "Draw"
            elif res == "Invalid":
                self.game_result = "Invalid Move"

    def draw_board(self):
        # Board Background
        pygame.draw.rect(
            self.screen, WHITE, (OFFSET_X, OFFSET_Y, BOARD_SIZE, BOARD_SIZE)
        )
        # Grid Lines
        for i in range(1, 3):
            pygame.draw.line(
                self.screen,
                BLACK,
                (OFFSET_X, OFFSET_Y + i * CELL_SIZE),
                (OFFSET_X + BOARD_SIZE, OFFSET_Y + i * CELL_SIZE),
                5,
            )
            pygame.draw.line(
                self.screen,
                BLACK,
                (OFFSET_X + i * CELL_SIZE, OFFSET_Y),
                (OFFSET_X + i * CELL_SIZE, OFFSET_Y + BOARD_SIZE),
                5,
            )

    def draw_pieces(self):
        for i in range(9):
            row = i // 3
            col = i % 3
            val = self.state[i]

            center_x = OFFSET_X + col * CELL_SIZE + CELL_SIZE // 2
            center_y = OFFSET_Y + row * CELL_SIZE + CELL_SIZE // 2

            if val == 1:  # X (AI)
                pygame.draw.line(
                    self.screen,
                    RED,
                    (center_x - 40, center_y - 40),
                    (center_x + 40, center_y + 40),
                    10,
                )
                pygame.draw.line(
                    self.screen,
                    RED,
                    (center_x + 40, center_y - 40),
                    (center_x - 40, center_y + 40),
                    10,
                )
            elif val == -1:  # O (Human)
                pygame.draw.circle(self.screen, BLUE, (center_x, center_y), 45, 8)

    def draw_dqn_viz(self):
        # Only visualize the AI's perspective (X) or Current Player?
        # Let's visualize the CURRENT TURN's perspective.

        viz_start_x = OFFSET_X + BOARD_SIZE + 50
        viz_start_y = OFFSET_Y
        viz_size = 300
        viz_cell = viz_size // 3

        # Label
        turn_p = self.env.current_player
        label = f"AI Analysis ({'X' if turn_p == 1 else 'O'} View)"
        title = self.font.render(label, True, WHITE)
        self.screen.blit(title, (viz_start_x, viz_start_y - 30))

        # Get Q values
        canonical_state = self.state * turn_p
        q_vals = self.agent.get_q_values(canonical_state)

        q_min, q_max = np.min(q_vals), np.max(q_vals)

        for i in range(9):
            row = i // 3
            col = i % 3
            val = q_vals[i]

            # Normalize color
            if q_max - q_min == 0:
                norm = 0.5
            else:
                norm = (val - q_min) / (q_max - q_min)

            r = int(255 * (1 - norm))
            g = int(255 * norm)
            color = (r, g, 50)

            rect = (
                viz_start_x + col * viz_cell,
                viz_start_y + row * viz_cell,
                viz_cell,
                viz_cell,
            )
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, BLACK, rect, 2)

            # Value
            val_text = self.font.render(f"{val:.1f}", True, BLACK)
            self.screen.blit(val_text, (rect[0] + 10, rect[1] + 30))

            # Ghost Piece Overlay
            if self.state[i] != 0:
                s = pygame.Surface((viz_cell, viz_cell))
                s.set_alpha(100)
                s.fill(BLACK)
                self.screen.blit(s, (rect[0], rect[1]))

    def draw_info(self):
        # Status Text
        status = (
            self.game_result
            if self.done
            else f"Turn: {'X (AI)' if self.env.current_player == 1 else 'O (YOU)'}"
        )
        color = RED if self.env.current_player == 1 else BLUE
        if self.done:
            color = GREEN

        s_text = self.big_font.render(status, True, color)
        self.screen.blit(s_text, (OFFSET_X, 20))

        # Stats Box
        stats_x = OFFSET_X + BOARD_SIZE + 50
        stats_y = OFFSET_Y + 320

        mode_str = self.mode
        if self.mode == "TRAIN_TURBO":
            mode_str += " (FAST)"

        lines = [
            f"Mode: {mode_str}",
            f"Total Episodes: {self.total_episodes}",
            f"Wins (AI-X): {self.wins_ai}",
            f"Wins (You-O): {self.wins_human}",
            f"Epsilon: {self.agent.epsilon:.3f}",
            f"Loss: {self.current_loss:.4f}",
        ]

        for i, line in enumerate(lines):
            c = YELLOW if "Mode" in line else WHITE
            t = self.font.render(line, True, c)
            self.screen.blit(t, (stats_x, stats_y + i * 25))


if __name__ == "__main__":
    app = GameApp()
    app.run()
