

# Interactive Q-Learning GridWorld Visualizer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Framework: PySide6](https://img.shields.io/badge/Framework-PySide6-blue)](https://www.qt.io/qt-for-python)
[![Library: PyTorch](https://img.shields.io/badge/Library-PyTorch-orange)](https://pytorch.org/)

An interactive desktop application built with Python, PySide6, and PyTorch that provides a real-time visualization of a Q-Learning agent learning to navigate a simple grid world.

This tool is designed for students, researchers, and enthusiasts to build an intuitive understanding of reinforcement learning concepts by observing an agent's behavior and tweaking its core hyperparameters on the fly.

---

## Features

-   **Real-time Agent Visualization:** Watch the agent explore the 5x5 grid and learn the optimal path to the goal.
-   **Live Performance Plot:** A dynamic graph tracks the agent's progress by plotting the number of steps taken per episode. A smoothed moving average is used to clearly visualize the learning trend.
-   **Interactive Hyperparameter Tuning:** Adjust the agent's key hyperparameters in real-time using sliders:
    -   **Learning Rate (α):** Controls the magnitude of Q-table updates.
    -   **Discount Factor (γ):** Determines the importance of future rewards.
    -   **Epsilon Decay:** Manages the rate of transition from exploration to exploitation.
-   **Informative Tooltips:** Hover over any slider to get a clear, concise explanation of what the hyperparameter does.
-   **Detailed Statistics:** View the current episode, moves within the episode, elapsed time, and the agent's exploration rate (epsilon).
-   **Experience Replay Monitoring:** A progress bar shows how full the agent's memory buffer is.
-   **Full Simulation Control:** Start, pause, and reset the training simulation at any time.
-   **Modern UI:** A clean, dark-themed interface built with the PySide6 (Qt for Python) framework.

## Core Concepts Demonstrated

This application serves as a practical demonstration of several fundamental reinforcement learning concepts:

-   **Q-Learning:** The core temporal-difference algorithm used by the agent to learn the value of state-action pairs.
-   **Exploration vs. Exploitation:** The role of the epsilon-greedy policy and the importance of decaying the exploration rate (ε) over time.
-   **Experience Replay:** The use of a memory buffer to store and sample past experiences, which stabilizes learning.
-   **Hyperparameter Impact:** The direct, observable effect of changing the learning rate, discount factor, and epsilon decay on the agent's speed and stability of learning.
-   **The Bellman Equation:** The Q-learning update rule is a practical application of the Bellman equation for estimating optimal Q-values.

## Tech Stack

-   **Language:** Python 3.8+
-   **GUI Framework:** PySide6 (The official Qt for Python project)
-   **RL Logic & Tensors:** PyTorch (for efficient Q-table operations)

## Installation and Setup

Follow these steps to get the application running on your local machine.

#### 1. Prerequisites
-   [Python 3.8](https://www.python.org/downloads/) or newer
-   [Git](https://git-scm.com/downloads/)

#### 2. Clone the Repository
Open your terminal and clone the repository:
```bash
git clone https://github.com/dovvnloading/Qlearning-grid-worldexplore
```

#### 3. Create a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

#### 4. Install Dependencies
Create a file named `requirements.txt` in the project's root directory with the following content:
```
torch
PySide6
```
Then, install the required packages using pip:
```bash
pip install -r requirements.txt
```
*(Note: This will install the CPU version of PyTorch, which is all that is needed for this project.)*

## How to Use

1.  **Run the Application:**
    Execute the main Python script from your terminal:
    ```bash
    python Qlearn grid world explorer.py
    ```

2.  **Start the Simulation:**
    Click the **"Start"** button to begin the agent's training.

3.  **Observe and Experiment:**
    -   Watch the blue agent square navigate the grid.
    -   Observe the "Learning Progress" plot. A downward curve indicates the agent is learning to reach the goal in fewer steps.
    -   Adjust the **Learning Rate**, **Discount Factor**, and **Epsilon Decay** sliders to see how they affect the agent's behavior and the shape of the learning curve.
    -   Use the **"Pause"** button to freeze the simulation for closer inspection and **"Reset"** to start the entire process over with a fresh agent.

## Code Overview

The project is structured into three main classes for modularity and clarity:

-   `QLearningAgent`: This class encapsulates all the reinforcement learning logic. It manages the Q-table, implements the epsilon-greedy policy, handles the environment steps, and performs learning via experience replay.
-   `ProgressCurveWidget`: A custom PySide6 widget responsible for rendering the performance plot. It takes a list of episode lengths and uses `QPainter` to draw a smoothed line graph.
-   `GridWorldUI (QMainWindow)`: The main application class. It constructs the UI, connects button signals to control slots, manages the training loop via a `QTimer`, and integrates the agent logic with the visual components.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
