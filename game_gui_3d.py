import pygame
import numpy as np
import sys
import math
from tictactoe_env import TicTacToeEnv
from dqn_agent import Agent

# --- CONSTANTS ---
WIDTH, HEIGHT = 1200, 700
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
DARK_BG = (20, 20, 25)
NEON_CYAN = (0, 255, 255)
NEON_MAGENTA = (255, 0, 255)

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


class NetworkVisualizer3D:
    def __init__(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)
        self.angle_y = 0
        self.angle_x = 0
        self.nodes = []  # List of (x, y, z, layer_index, value)
        self.connections = []

        # Layout configuration
        self.layer_depths = [
            -150,
            -50,
            50,
            150,
        ]  # Z positions for Input, H1, H2, Output
        self.focal_length = 400
        self.center_x = x + w // 2
        self.center_y = y + h // 2

        # Cache structure
        self.structure_generated = False

    def generate_structure(self):
        self.nodes = []

        # Layer 0: Input (9 neurons) -> 3x3 Grid
        for i in range(9):
            r, c = i // 3, i % 3
            # Centered grid
            x = (c - 1) * 30
            y = (r - 1) * 30
            z = self.layer_depths[0]
            self.nodes.append({"pos": [x, y, z], "layer": 0, "val": 0.0, "id": i})

        # Layer 1: Hidden (128 neurons) -> 8x16 Grid
        # Visualizing 128 is a lot, let's condense visual representation or show them all small
        for i in range(128):
            r, c = i // 16, i % 16  # 8 rows, 16 cols
            x = (c - 7.5) * 15
            y = (r - 3.5) * 15
            z = self.layer_depths[1]
            self.nodes.append({"pos": [x, y, z], "layer": 1, "val": 0.0, "id": i})

        # Layer 2: Hidden (128 neurons) -> 8x16 Grid
        for i in range(128):
            r, c = i // 16, i % 16
            x = (c - 7.5) * 15
            y = (r - 3.5) * 15
            z = self.layer_depths[2]
            self.nodes.append({"pos": [x, y, z], "layer": 2, "val": 0.0, "id": i})

        # Layer 3: Output (9 neurons) -> 3x3 Grid
        for i in range(9):
            r, c = i // 3, i % 3
            x = (c - 1) * 30
            y = (r - 1) * 30
            z = self.layer_depths[3]
            self.nodes.append({"pos": [x, y, z], "layer": 3, "val": 0.0, "id": i})

        self.structure_generated = True

    def update_activations(self, activations):
        # Activations is list of arrays: [Input(9), H1(128), H2(128), Out(9)]
        idx = 0
        for layer_vals in activations:
            # Normalize layer for display (simple min-max or abs scaling)
            max_val = np.max(np.abs(layer_vals)) if len(layer_vals) > 0 else 1
            if max_val == 0:
                max_val = 1

            for val in layer_vals:
                if idx < len(self.nodes):
                    self.nodes[idx]["val"] = val / max_val  # Scale -1 to 1 roughly
                    idx += 1

    def rotate(self, dx, dy):
        self.angle_y += dx * 0.01
        self.angle_x += dy * 0.01

    def project(self, x, y, z):
        # Rotation Y
        qy = y
        qx = x * math.cos(self.angle_y) - z * math.sin(self.angle_y)
        qz = x * math.sin(self.angle_y) + z * math.cos(self.angle_y)

        # Rotation X
        y_rot = qy * math.cos(self.angle_x) - qz * math.sin(self.angle_x)
        z_rot = qy * math.sin(self.angle_x) + qz * math.cos(self.angle_x)

        # Perspective Project
        # Move object forward so it's in front of camera
        z_cam = z_rot + 400
        if z_cam == 0:
            z_cam = 0.1

        scale = self.focal_length / z_cam
        x_2d = qx * scale + self.center_x
        y_2d = y_rot * scale + self.center_y

        return int(x_2d), int(y_2d), scale, z_rot

    def draw(self, screen):
        if not self.structure_generated:
            self.generate_structure()

        # Draw Background for Viz
        pygame.draw.rect(screen, (10, 10, 15), self.rect)
        pygame.draw.rect(screen, (50, 50, 60), self.rect, 2)

        # Project all points
        projected_nodes = []
        for node in self.nodes:
            px, py, scale, z_depth = self.project(*node["pos"])
            projected_nodes.append((px, py, scale, z_depth, node))

        # Sort by depth (painter's algorithm)
        projected_nodes.sort(key=lambda x: x[3], reverse=True)

        # Draw connections (only highly active ones to save FPS, or just layer-to-layer lines)
        # Drawing all connections (128*128) is too heavy for PyGame.
        # We will draw "Representative" flow lines based on layer depth

        # Draw Nodes
        for px, py, scale, z, node in projected_nodes:
            if not self.rect.collidepoint(px, py):
                continue

            val = node["val"]
            # Color based on activation
            # Negative = Blue, Positive = Red, Near Zero = Dark
            intensity = min(255, int(abs(val) * 255))
            if val > 0:
                color = (intensity, intensity // 4, intensity // 4)  # Reddish
            else:
                color = (intensity // 4, intensity // 4, intensity)  # Bluish

            if intensity < 20:
                color = (30, 30, 40)  # Base color

            size = int(3 * scale)
            if node["layer"] == 0 or node["layer"] == 3:
                size = int(6 * scale)  # Make Input/Output bigger

            pygame.draw.circle(screen, color, (px, py), size)

            # Glow effect for active neurons
            if intensity > 100:
                pygame.draw.circle(screen, color, (px, py), size + 2, 1)

        # Draw Labels
        font = pygame.font.SysFont("Arial", 14)
        labels = ["Input", "Hidden 1", "Hidden 2", "Output"]
        for i, label in enumerate(labels):
            # Project a point above the layer
            lx, ly, _, _ = self.project(0, -80, self.layer_depths[i])
            if self.rect.collidepoint(lx, ly):
                txt = font.render(label, True, WHITE)
                screen.blit(txt, (lx - txt.get_width() // 2, ly))


class GameApp:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Tic Tac Toe DQN - 3D Neural Visualizer")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 20)
        self.big_font = pygame.font.SysFont("Arial", 48)

        # Initialize Game Components
        self.env = TicTacToeEnv()
        self.agent = Agent(state_dim=9, action_dim=9)
        if self.agent.load(MODEL_PATH):
            print("Loaded existing model.")

        self.human_side = -1
        self.ai_side = 1
        self.mode = "PLAY"

        # Stats
        self.total_episodes = 0
        self.wins_ai = 0
        self.wins_human = 0
        self.current_loss = 0

        # UI Buttons
        btn_y = HEIGHT - 60
        self.buttons = [
            Button(50, btn_y, 140, 40, "Restart", "RESET"),
            Button(200, btn_y, 140, 40, "Watch Train", "TRAIN_WATCH"),
            Button(350, btn_y, 140, 40, "Turbo Train", "TRAIN_TURBO"),
            Button(500, btn_y, 140, 40, "Play Mode", "PLAY"),
            Button(650, btn_y, 140, 40, "Save Model", "SAVE"),
        ]

        # 3D Visualizer Instance
        # Placed on right side
        self.viz3d = NetworkVisualizer3D(OFFSET_X + BOARD_SIZE + 50, 80, 600, 500)

        self.dragging = False
        self.last_mouse = (0, 0)

        self.reset_game()

    def reset_game(self):
        self.state = self.env.reset()
        self.done = False
        self.game_result = ""
        if self.mode == "PLAY" and self.env.current_player == self.ai_side:
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

                    # Handle 3D Rotation
                    if self.dragging:
                        dx = mouse_pos[0] - self.last_mouse[0]
                        dy = mouse_pos[1] - self.last_mouse[1]
                        self.viz3d.rotate(dx, dy)
                        self.last_mouse = mouse_pos

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left Click
                        # Buttons
                        for btn in self.buttons:
                            if btn.is_clicked(mouse_pos):
                                self.handle_button(btn.action_code)

                        # Board
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

                        # 3D Viz Drag Start
                        if self.viz3d.rect.collidepoint(mouse_pos):
                            self.dragging = True
                            self.last_mouse = mouse_pos

                elif event.type == pygame.MOUSEBUTTONUP:
                    self.dragging = False

            # --- LOGIC ---
            if self.mode == "TRAIN_WATCH":
                if self.done:
                    self.reset_and_track_train()
                else:
                    self.step_ai_logic(eval_mode=False)

            elif self.mode == "TRAIN_TURBO":
                steps = 0
                while (
                    steps < 200
                ):  # Reduced turbo slightly to allow 3D rendering breathing room
                    if self.done:
                        self.reset_and_track_train()
                    self.step_ai_logic(eval_mode=False)
                    steps += 1

            elif self.mode == "PLAY":
                if not self.done and self.env.current_player == self.ai_side:
                    self.step_ai_logic(eval_mode=True)

            # --- UPDATE 3D VIZ DATA ---
            # Get activations for the CURRENT state from the CURRENT player's perspective
            turn_player = self.env.current_player
            canonical_state = self.state * turn_player
            acts = self.agent.get_activations(canonical_state)
            self.viz3d.update_activations(acts)

            # --- DRAWING ---
            self.draw_board()
            self.draw_pieces()
            self.draw_info()
            self.viz3d.draw(self.screen)

            for btn in self.buttons:
                btn.draw(self.screen, self.font)

            # Draw 3D Instructions
            inst = self.font.render("Drag mouse here to rotate 3D Brain", True, GRAY)
            self.screen.blit(
                inst,
                (
                    self.viz3d.rect.centerx - inst.get_width() // 2,
                    self.viz3d.rect.bottom + 10,
                ),
            )

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()

    def handle_button(self, code):
        if code == "RESET":
            self.reset_game()
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
            print("Model Saved.")

    def reset_and_track_train(self):
        self.total_episodes += 1
        if self.total_episodes % 100 == 0:
            self.agent.update_target_network()
        self.state = self.env.reset()
        self.done = False

    def step_human(self, idx):
        if self.state[idx] == 0:
            self.execute_move(idx)

    def step_ai_logic(self, eval_mode=True):
        turn_player = self.env.current_player
        canonical_state = self.state * turn_player
        action = self.agent.act(canonical_state, eval_mode=eval_mode)

        if self.state[action] != 0:
            valid_indices = [i for i, x in enumerate(self.state) if x == 0]
            if valid_indices:
                action = np.random.choice(valid_indices)
            else:
                return

        self.execute_move(action)

    def execute_move(self, action_idx):
        turn_player = self.env.current_player
        canonical_state = self.state * turn_player
        next_state, reward, done, info = self.env.step(action_idx)
        canonical_next_state = next_state * turn_player

        self.agent.remember(
            canonical_state, action_idx, reward, canonical_next_state, done
        )
        loss = self.agent.replay(64)
        if loss:
            self.current_loss = loss

        self.state = next_state
        self.done = done
        if done:
            res = info["result"]
            if res == "Win":
                if turn_player == 1:
                    self.wins_ai += 1
                else:
                    self.wins_human += 1
                self.game_result = f"Winner: {'AI' if turn_player == 1 else 'Human'}"
            elif res == "Draw":
                self.game_result = "Draw"
            else:
                self.game_result = "Invalid"

    def draw_board(self):
        # Board Background
        pygame.draw.rect(
            self.screen, WHITE, (OFFSET_X, OFFSET_Y, BOARD_SIZE, BOARD_SIZE)
        )
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
            row, col = i // 3, i % 3
            val = self.state[i]
            cx = OFFSET_X + col * CELL_SIZE + CELL_SIZE // 2
            cy = OFFSET_Y + row * CELL_SIZE + CELL_SIZE // 2
            if val == 1:
                pygame.draw.line(
                    self.screen, RED, (cx - 40, cy - 40), (cx + 40, cy + 40), 10
                )
                pygame.draw.line(
                    self.screen, RED, (cx + 40, cy - 40), (cx - 40, cy + 40), 10
                )
            elif val == -1:
                pygame.draw.circle(self.screen, BLUE, (cx, cy), 45, 8)

    def draw_info(self):
        status = (
            self.game_result
            if self.done
            else f"Turn: {'AI (X)' if self.env.current_player == 1 else 'YOU (O)'}"
        )
        color = RED if self.env.current_player == 1 else BLUE
        if self.done:
            color = GREEN
        self.screen.blit(self.big_font.render(status, True, color), (OFFSET_X, 20))

        stats_x = OFFSET_X
        stats_y = OFFSET_Y + BOARD_SIZE + 30
        lines = [
            f"Mode: {self.mode}",
            f"Episodes: {self.total_episodes}",
            f"AI Wins: {self.wins_ai} | Human Wins: {self.wins_human}",
            f"Epsilon: {self.agent.epsilon:.3f} | Loss: {self.current_loss:.4f}",
        ]
        for i, line in enumerate(lines):
            self.screen.blit(
                self.font.render(line, True, WHITE), (stats_x, stats_y + i * 25)
            )


if __name__ == "__main__":
    app = GameApp()
    app.run()
