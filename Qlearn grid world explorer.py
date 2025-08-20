import sys
import random
import time
import torch
from collections import deque

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QGraphicsScene, QGraphicsView, QGraphicsRectItem,
    QPushButton, QProgressBar, QSlider, QGroupBox, QGridLayout
)
from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import QColor, QPalette, QPainter, QPen, QFontMetrics


# ----------------------------------------
#          Progress Plotting Widget
# ----------------------------------------
class ProgressCurveWidget(QWidget):
    """
    A custom QWidget to render a real-time plot of the agent's learning progress,
    displaying the number of steps taken per episode. It includes a moving
    average to smooth the curve for better trend visualization.
    """
    def __init__(self, parent=None):
        """Initializes the plot widget."""
        super().__init__(parent)
        self.setMinimumHeight(150)
        self._data = []
        self._smoothed_data = []

    def update_data(self, new_data: list):
        """
        Updates the plot with new data and triggers a repaint.

        Args:
            new_data: A list containing the number of steps for each completed episode.
        """
        self._data = new_data
        # A moving average makes the learning trend much easier to see.
        self._smoothed_data = self._moving_average(self._data, window_size=50)
        self.update()  # Request a repaint of the widget.

    def _moving_average(self, data: list, window_size: int) -> list:
        """Calculates a simple moving average for the given data."""
        if not data:
            return []
        return [
            sum(data[max(0, i - window_size):i + 1]) / len(data[max(0, i - window_size):i + 1])
            for i in range(len(data))
        ]

    def paintEvent(self, event):
        """
        Handles the painting of the widget, drawing the axes and the data curve.
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), self.palette().color(QPalette.Base))

        if not self._smoothed_data:
            return  # Don't draw if there is no data

        # Define margins for axes and labels.
        margin_top, margin_bottom, margin_left, margin_right = 10, 20, 30, 10
        graph_width = self.width() - margin_left - margin_right
        graph_height = self.height() - margin_top - margin_bottom

        # Determine the data range for scaling. Use raw data for max Y to ensure
        # the smoothed curve never goes out of bounds.
        max_y = max(1, max(self._data))
        max_x = len(self._smoothed_data) - 1
        if max_x <= 0: max_x = 1 # Avoid division by zero

        # Calculate scaling factors to map data points to widget coordinates.
        x_scale = graph_width / max_x
        y_scale = graph_height / max_y

        # --- Draw Axes and Labels ---
        axis_pen = QPen(self.palette().color(QPalette.AlternateBase))
        painter.setPen(axis_pen)
        painter.drawLine(margin_left, margin_top, margin_left, self.height() - margin_bottom) # Y-axis
        painter.drawLine(margin_left, self.height() - margin_bottom, self.width() - margin_right, self.height() - margin_bottom) # X-axis

        text_pen = QPen(self.palette().color(QPalette.Text))
        painter.setPen(text_pen)
        fm = QFontMetrics(self.font())
        painter.drawText(0, margin_top + fm.ascent() // 2, f"{int(max_y)}") # Y-axis max label
        x_label = f"{len(self._smoothed_data)}"
        painter.drawText(self.width() - margin_right - fm.horizontalAdvance(x_label), self.height(), x_label) # X-axis max label

        # --- Draw Smoothed Learning Curve ---
        curve_pen = QPen(QColor(90, 140, 255), 2)  # A bright blue for visibility
        painter.setPen(curve_pen)
        
        # Convert data points to QPointF screen coordinates.
        # Note: Y-axis is inverted in screen coordinates (0 is at the top).
        points = [
            QPointF(
                margin_left + i * x_scale,
                (self.height() - margin_bottom) - (val * y_scale)
            ) for i, val in enumerate(self._smoothed_data)
        ]

        if len(points) > 1:
            painter.drawPolyline(points)


# ----------------------------------------
#          Q-Learning Agent
# ----------------------------------------
class QLearningAgent:
    """
    Implements a Q-Learning agent with experience replay.

    The agent learns to navigate a grid world to reach a goal state.
    It uses a Q-table to store state-action values and a memory buffer
    (experience replay) to learn from batches of past experiences, which
    stabilizes the learning process.
    """
    def __init__(self, n_states=25, n_actions=8, start_state=0, goal_state=24):
        """Initializes the agent and its hyperparameters."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_states = n_states
        self.n_actions = n_actions
        self.start_state = start_state
        self.goal_state = goal_state

        # Hyperparameters
        self.learning_rate = 0.5      # α: How much to update Q-values.
        self.discount_factor = 0.95   # γ: Importance of future rewards.
        self.epsilon = 1.0            # ε: Initial exploration rate.
        self.epsilon_end = 0.1        # Minimum exploration rate.
        self.epsilon_decay = 0.995    # Rate at which ε decreases.
        
        # Experience Replay
        self.batch_size = 128
        self.memory_size = 50000
        # deque is highly efficient for appends and pops from either end.
        self.memory = deque(maxlen=self.memory_size)
        
        # Q-table implemented with a PyTorch tensor for potential GPU acceleration.
        self.Q = torch.zeros(n_states, n_actions, device=self.device)

    @staticmethod
    def state_to_xy(state: int) -> tuple[int, int]:
        """Converts a 1D state index to 2D grid coordinates."""
        return state % 5, state // 5

    def step(self, state: int, action: int) -> tuple[int, float]:
        """
        Simulates the agent taking an action in the environment.

        Args:
            state: The current state of the agent.
            action: The action to be taken.

        Returns:
            A tuple containing (next_state, reward).
        """
        x, y = self.state_to_xy(state)
        if action == 0: y = max(0, y - 1)    # Up
        elif action == 1: y = min(4, y + 1)    # Down
        elif action == 2: x = max(0, x - 1)    # Left
        elif action == 3: x = min(4, x + 1)    # Right
        elif action == 4: x, y = max(0, x - 1), max(0, y - 1) # Up-Left
        elif action == 5: x, y = min(4, x + 1), max(0, y - 1) # Up-Right
        elif action == 6: x, y = max(0, x - 1), min(4, y + 1) # Down-Left
        elif action == 7: x, y = min(4, x + 1), min(4, y + 1) # Down-Right

        next_state = y * 5 + x
        reward = 1.0 if next_state == self.goal_state else 0.0
        return next_state, reward

    def choose_action(self, state: int) -> int:
        """
        Selects an action using an epsilon-greedy policy.

        With probability epsilon, a random action is chosen (exploration).
        Otherwise, the action with the highest Q-value is chosen (exploitation).
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return torch.argmax(self.Q[state]).item()

    def remember(self, state, action, reward, next_state, done):
        """Stores an experience tuple in the replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """
        Trains the agent by sampling a batch of experiences from memory.
        This method implements the core Q-learning update rule using batch
        processing for efficiency.
        """
        if len(self.memory) < self.batch_size:
            return # Don't train until we have enough experiences.

        # Sample a random batch of transitions from memory.
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to PyTorch tensors for vectorized operations.
        states = torch.tensor(states, device=self.device, dtype=torch.long)
        actions = torch.tensor(actions, device=self.device, dtype=torch.long)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.long)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float32)

        # Q-learning update rule: Q(s,a) <- Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
        # We compute the target values for the batch first.
        max_next_Q = torch.max(self.Q[next_states], dim=1)[0]
        targets = rewards + self.discount_factor * (1.0 - dones) * max_next_Q
        
        # Get the Q-values for the specific actions taken.
        Q_values = self.Q[states, actions]
        
        # Update the Q-table using the batch update.
        self.Q[states, actions] += self.learning_rate * (targets - Q_values)

    def decay_epsilon(self):
        """Reduces the exploration rate (epsilon) over time."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


# ----------------------------------------
#          Main UI Application
# ----------------------------------------
class GridWorldUI(QMainWindow):
    """
    The main window for the Q-Learning GridWorld visualization application.

    It handles the UI layout, user controls (start, pause, reset),
    hyperparameter sliders, and orchestrates the training loop via a QTimer.
    """
    def __init__(self):
        """Initializes the main window and all UI components."""
        super().__init__()
        self.setWindowTitle("Q-Learning GridWorld")

        # --- Initialize state and agent ---
        self.agent = QLearningAgent()
        self.state = self.agent.start_state
        self.episode = 0
        self.moves = 0
        self.total_moves = 0
        self.episode_steps = []  # Stores steps per episode for plotting.
        self.start_time = time.time()
        
        # --- Build UI and set up timer ---
        self._apply_dark_theme()
        self._build_ui()
        
        # Let the layout system calculate the optimal window size.
        self.adjustSize()

        # The QTimer drives the simulation steps.
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._train_step)

    def _apply_dark_theme(self):
        """Sets a dark color palette for the application."""
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(24, 24, 28))
        palette.setColor(QPalette.Base, QColor(28, 28, 34))
        palette.setColor(QPalette.AlternateBase, QColor(36, 36, 44))
        palette.setColor(QPalette.Button, QColor(36, 36, 44))
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.Highlight, QColor(90, 140, 255))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(palette)

    def _build_ui(self):
        """Constructs all UI widgets and layouts."""
        main_layout = QHBoxLayout()
        side_layout = QVBoxLayout()

        # --- Graphics Scene and View ---
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setFixedSize(520, 520)
        self.view.setBackgroundBrush(QColor(18, 18, 22))
        self.view.setRenderHint(QPainter.Antialiasing, True)
        self.grid_size = 100
        
        grid_pen = QPen(QColor(90, 90, 100), 1)
        for i in range(5):
            for j in range(5):
                rect = QGraphicsRectItem(i * self.grid_size, j * self.grid_size, self.grid_size, self.grid_size)
                rect.setPen(grid_pen)
                self.scene.addItem(rect)

        # --- Goal and Agent Graphics Items ---
        gx, gy = self.agent.state_to_xy(self.agent.goal_state)
        self.goal_item = QGraphicsRectItem(gx * self.grid_size, gy * self.grid_size, self.grid_size, self.grid_size)
        self.goal_item.setBrush(QColor(0, 180, 120))
        self.goal_item.setPen(QPen(Qt.NoPen))
        self.scene.addItem(self.goal_item)

        self.agent_item = QGraphicsRectItem(0, 0, self.grid_size - 40, self.grid_size - 40)
        self.agent_item.setBrush(QColor(100, 150, 255))
        self.agent_item.setPen(QPen(Qt.NoPen))
        self.scene.addItem(self.agent_item)
        self._reposition_agent()

        # --- Sidebar Widgets ---
        self.lbl_episode = QLabel("Episode: 0")
        self.lbl_moves = QLabel("Moves: 0")
        self.lbl_epsilon = QLabel("Epsilon: 1.00")
        self.lbl_time = QLabel("Elapsed: 00:00")
        self.memory_bar = QProgressBar()
        self.memory_bar.setRange(0, self.agent.memory_size)
        self.memory_bar.setTextVisible(True)
        self.btn_start = QPushButton("Start")
        self.btn_pause = QPushButton("Pause")
        self.btn_reset = QPushButton("Reset")
        self.progress_curve = ProgressCurveWidget()

        # --- Hyperparameter Controls GroupBox ---
        hyper_group_box = QGroupBox("Hyperparameters")
        hyper_layout = QGridLayout()

        # Helper function to create a slider with labels.
        def create_slider(label, min_val, max_val, initial_val, callback, tooltip):
            slider = QSlider(Qt.Horizontal)
            slider.setRange(min_val, max_val)
            slider.setValue(initial_val)
            slider.valueChanged.connect(callback)
            slider.setToolTip(tooltip)
            value_label = QLabel()
            hyper_layout.addWidget(QLabel(label), hyper_layout.rowCount(), 0)
            hyper_layout.addWidget(slider, hyper_layout.rowCount()-1, 1)
            hyper_layout.addWidget(value_label, hyper_layout.rowCount()-1, 2)
            return slider, value_label

        self.lr_slider, self.lr_label_val = create_slider(
            "Learning Rate:", 0, 1000, int(self.agent.learning_rate * 1000), self._update_learning_rate,
            "Learning Rate (α): Controls how much to change Q-values on each update.\nHigher values learn faster but can be unstable.")
        
        self.df_slider, self.df_label_val = create_slider(
            "Discount Factor:", 0, 1000, int(self.agent.discount_factor * 1000), self._update_discount_factor,
            "Discount Factor (γ): Determines the importance of future rewards.\nCloser to 1.0 makes the agent more 'farsighted'.")
            
        self.ed_slider, self.ed_label_val = create_slider(
            "Epsilon Decay:", 9900, 10000, int(self.agent.epsilon_decay * 10000), self._update_epsilon_decay,
            "Epsilon Decay: Rate at which exploration (randomness) decreases.\nHigher values (closer to 1.0) mean slower decay.")
        
        hyper_group_box.setLayout(hyper_layout)
        # Manually trigger initial update to set label text.
        self._update_learning_rate(self.lr_slider.value())
        self._update_discount_factor(self.df_slider.value())
        self._update_epsilon_decay(self.ed_slider.value())

        # --- Assemble Layouts ---
        side_layout.addWidget(self.lbl_episode)
        side_layout.addWidget(self.lbl_moves)
        side_layout.addWidget(self.lbl_epsilon)
        side_layout.addWidget(self.lbl_time)
        side_layout.addSpacing(8)
        side_layout.addWidget(QLabel("Memory Usage:"))
        side_layout.addWidget(self.memory_bar)
        side_layout.addSpacing(12)
        side_layout.addWidget(self.btn_start)
        side_layout.addWidget(self.btn_pause)
        side_layout.addWidget(self.btn_reset)
        side_layout.addWidget(hyper_group_box)
        side_layout.addStretch() # Pushes plot to the bottom.
        side_layout.addWidget(QLabel("Learning Progress (Smoothed Steps/Episode):"))
        side_layout.addWidget(self.progress_curve)
        
        central = QWidget()
        main_layout.addWidget(self.view)
        main_layout.addLayout(side_layout)
        central.setLayout(main_layout)
        self.setCentralWidget(central)
        
        # --- Connect Signals and Slots ---
        self.btn_start.clicked.connect(self._start)
        self.btn_pause.clicked.connect(self._pause)
        self.btn_reset.clicked.connect(self._reset)

    # --- Slot methods for sliders ---
    def _update_learning_rate(self, value):
        self.agent.learning_rate = value / 1000.0
        self.lr_label_val.setText(f"{self.agent.learning_rate:.3f}")

    def _update_discount_factor(self, value):
        self.agent.discount_factor = value / 1000.0
        self.df_label_val.setText(f"{self.agent.discount_factor:.3f}")

    def _update_epsilon_decay(self, value):
        self.agent.epsilon_decay = value / 10000.0
        self.ed_label_val.setText(f"{self.agent.epsilon_decay:.4f}")

    # --- Control methods ---
    def _start(self):
        """Starts or resumes the training simulation."""
        if not self.timer.isActive():
            self.timer.start(10)  # Interval in ms. Lower is faster.
    
    def _pause(self):
        """Pauses the training simulation."""
        self.timer.stop()

    def _reset(self):
        """Resets the entire simulation to its initial state."""
        self.timer.stop()
        self.agent = QLearningAgent()
        self.state = self.agent.start_state
        self.episode = 0
        self.moves = 0
        self.total_moves = 0
        self.episode_steps = []
        self.start_time = time.time()
        
        # Reset UI elements
        self.memory_bar.setMaximum(self.agent.memory_size)
        self.progress_curve.update_data(self.episode_steps)
        self.lr_slider.setValue(int(self.agent.learning_rate * 1000))
        self.df_slider.setValue(int(self.agent.discount_factor * 1000))
        self.ed_slider.setValue(int(self.agent.epsilon_decay * 10000))
        
        self._reposition_agent()
        self._update_sidebar()

    # --- Core Training and UI Update Logic ---
    def _train_step(self):
        """Executes a single step of the agent's training loop."""
        # 1. Agent chooses an action.
        action = self.agent.choose_action(self.state)
        # 2. Agent performs the action in the environment.
        next_state, reward = self.agent.step(self.state, action)
        done = (next_state == self.agent.goal_state)
        # 3. Agent stores this experience in memory.
        self.agent.remember(self.state, action, reward, next_state, done)
        # 4. Agent learns from a batch of past experiences.
        self.agent.replay()
        
        # 5. Update state and counters.
        self.state = next_state
        self.moves += 1
        self.total_moves += 1

        # 6. Check for episode termination.
        if done:
            self.episode += 1
            self.agent.decay_epsilon()
            self.episode_steps.append(self.moves)
            self.state = self.agent.start_state # Reset agent to start

            # Update plot periodically to avoid excessive redrawing, which can slow down the UI.
            if self.episode % 10 == 0:
                self.progress_curve.update_data(self.episode_steps)

            self.moves = 0

        # 7. Update the UI.
        self._reposition_agent()
        self._update_sidebar()

    def _reposition_agent(self):
        """Updates the visual position of the agent on the grid."""
        x, y = self.agent.state_to_xy(self.state)
        self.agent_item.setRect(
            x * self.grid_size + 20, y * self.grid_size + 20,
            self.grid_size - 40, self.grid_size - 40
        )

    def _update_sidebar(self):
        """Refreshes all the text labels and progress bars in the sidebar."""
        self.lbl_episode.setText(f"Episode: {self.episode}")
        self.lbl_moves.setText(f"Moves: {self.moves}")
        self.lbl_epsilon.setText(f"Epsilon: {self.agent.epsilon:.2f}")
        self.memory_bar.setValue(len(self.agent.memory))
        elapsed = int(time.time() - self.start_time)
        self.lbl_time.setText(f"Elapsed: {elapsed // 60:02d}:{elapsed % 60:02d}")


if __name__ == "__main__":
    """The main entry point for the application."""
    app = QApplication(sys.argv)
    window = GridWorldUI()
    window.show()
    sys.exit(app.exec())
