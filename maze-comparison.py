import numpy as np
import pygame
import time
import random

# Maze map definition
maze = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
])

# Define maze dimensions
maze_height, maze_width = maze.shape
init_position = (1, 1)  # Starting position of the agent
end_position = (maze_height - 2, maze_width - 2)  # Ending position in the maze

# Q-Learning parameters
num_episodes = 2000  # Number of training episodes
epsilon = 0.1  # Exploration rate
learning_rate = 0.1  # Learning rate for Q-learning
discount_factor = 0.9  # Discount factor for future rewards

# Initialize Q-Table with zeros
Q_table = np.zeros((maze_height, maze_width, 4))

# Action mapping
actions = ["up", "down", "left", "right"]  # Action names in Chinese (up, down, left, right)
action_map = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Corresponding movements for actions


class Agent:
    def __init__(self, trained=False):
        self.state = init_position  # Initialize agent's state
        self.trained = trained  # Whether the agent is trained or not

    def choose_action(self):
        # Choose action based on whether the agent is trained
        if self.trained:
            return np.argmax(Q_table[self.state[0], self.state[1], :])  # Choose optimal action
        else:
            return random.choice(range(4))  # Randomly choose an action

    def update_state(self, action):
        # Update agent's position based on the chosen action
        row, col = self.state
        dr, dc = action_map[action]  # Get change in row and column based on action
        new_row, new_col = row + dr, col + dc  # Calculate new position
        if maze[new_row, new_col] == 0:  # Check if new position is a valid move
            self.state = (new_row, new_col)  # Update state to new position


def q_learning():
    # Core Q-Learning algorithm
    for episode in range(num_episodes):
        state = init_position  # Start each episode from the initial position
        while state != end_position:  # Continue until the agent reaches the end position
            # Choose action based on epsilon-greedy strategy
            action = np.random.choice(4) if np.random.uniform(0, 1) < epsilon else np.argmax(Q_table[state[0], state[1], :])
            new_row, new_col = state[0] + action_map[action][0], state[1] + action_map[action][1]  # Calculate new position
            if maze[new_row, new_col] == 1:  # Check for wall collision
                reward = -100  # Penalize for hitting a wall
                new_state = state  # Stay in the same state
            else:
                reward = -1  # Penalize for each step taken
                new_state = (new_row, new_col)  # Update to new valid state

            # Update Q-value using the Q-learning formula
            Q_table[state[0], state[1], action] += learning_rate * (
                reward + discount_factor * np.max(Q_table[new_state[0], new_state[1], :]) -
                Q_table[state[0], state[1], action]
            )
            state = new_state  # Update state for the next iteration


def draw_maze(screen, offset_x):
    # Draw the maze on the screen
    for row in range(maze_height):
        for col in range(maze_width):
            color = (255, 255, 255) if maze[row, col] == 0 else (0, 0, 0)  # White for path, black for walls
            pygame.draw.rect(screen, color, (offset_x + col * cell_size, row * cell_size, cell_size, cell_size))


def draw_agent(screen, agent, color, offset_x):
    # Draw the agent on the screen
    row, col = agent.state
    pygame.draw.circle(screen, color, (offset_x + col * cell_size + cell_size // 2, row * cell_size + cell_size // 2), cell_size // 3)


# Train the Q-Learning agent
q_learning()

# Pygame initialization
pygame.init()
cell_size = 40  # Size of each cell in the maze
screen_width = maze_width * cell_size * 2  # Double width for side-by-side comparison
screen_height = maze_height * cell_size  # Height of the maze
screen = pygame.display.set_mode((screen_width, screen_height))  # Create display window
pygame.display.set_caption("Maze Learning Comparison")  # Set window title

# Create agents
random_agent = Agent(trained=False)  # Agent before training (random actions)
trained_agent = Agent(trained=True)  # Agent after training (Q-learning)

running = True
clock = pygame.time.Clock()

# Run the agents
while running:
    screen.fill((0, 0, 0))  # Clear the screen

    # Draw the maze (left - random agent, right - Q-learning agent)
    draw_maze(screen, 0)
    draw_maze(screen, maze_width * cell_size)

    # Draw the agents (red - before training, blue - after training)
    draw_agent(screen, random_agent, (255, 0, 0), 0)  # Random agent in red
    draw_agent(screen, trained_agent, (0, 0, 255), maze_width * cell_size)  # Trained agent in blue

    pygame.display.flip()  # Update the display
    time.sleep(0.3)  # Delay for visualization

    # Update the state of the random agent
    if random_agent.state != end_position:
        random_action = random_agent.choose_action()  # Choose a random action
        random_agent.update_state(random_action)  # Update agent's position

    # Update the state of the trained agent
    if trained_agent.state != end_position:
        trained_action = trained_agent.choose_action()  # Choose an action based on learned Q-values
        trained_agent.update_state(trained_action)  # Update agent's position

    # If both agents reach the end position, exit the loop
    if random_agent.state == end_position and trained_agent.state == end_position:
        time.sleep(1)  # Pause for a moment before closing
        running = False

pygame.quit()  # Close the Pygame window