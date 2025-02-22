import numpy as np
import random
import pygame
import time
import os

epsilon = 0.1  # Randomness rate for exploration
learning_rate = 0.1  # Learning rate for Q-learning
discount_factor = 0.9  # Discount factor for future rewards
num_episodes = 2000  # Number of iterations

# Make a 10x10 Maze
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

# Clear console output
def clear_console():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

# Maze dimensions
maze_height, maze_width = maze.shape

init_position = {"x": 1, "y": 1}  # Start
end_position = {"x": maze_height - 2, "y": maze_width - 2}  # End

# Maze interface
def draw_maze():
    for row in range(maze_height):
        for col in range(maze_width):
            color = (255, 255, 255) if maze[row, col] == 0 else (0, 0, 0)  # White for path, black for walls
            pygame.draw.rect(screen, color, (col * cell_size, row * cell_size, cell_size, cell_size))

# AI Agent class
class Agent:
    def __init__(self, state, actions):
        self.state = state  # Current state (position of the agent)
        self.actions = actions  # Possible actions

    def choose_action(self, epsilon): # Choose action based on epsilon-greedy strategy
        if np.random.uniform(0, 1) < epsilon:
            return np.random.choice(self.actions)  # Explore
        else:
            return np.argmax(Q_table[self.state[0], self.state[1], :])  # Exploit

    def update_state(self, new_state):
        self.state = new_state  # Update agent's state

# Console drawing function
def draw_agent_console(agent):
    row, col = agent.state
    maze_with_agent = maze.copy()  # Create a copy of the maze
    maze_with_agent[row, col] = 2  # Use number 2 to represent agent's position
    for row in range(maze_height):
        for col in range(maze_width):
            char = "#" if maze_with_agent[row, col] == 1 else "A" if maze_with_agent[row, col] == 2 else " "
            print(char, end=" ")  # Print maze with agent
        print()

# Draw the agent in the maze
def draw_agent(agent):
    row, col = agent.state
    pygame.draw.circle(screen, (255, 0, 0), (col * cell_size + cell_size // 2, row * cell_size + cell_size // 2), cell_size // 3)


def show_all():
    # Display the maze and agent in pygame
    draw_maze()
    draw_agent(agent)
    # Display the maze and agent in console
    clear_console()
    draw_agent_console(agent)

# Update the agent's position in the maze
def update_agent(agent, action):
    row, col = agent.state
    if action == 0:  # Move up
        row = max(row - 1, 0)
    elif action == 1:  # Move down
        row = min(row + 1, maze_height - 1)
    elif action == 2:  # Move left
        col = max(col - 1, 0)
    else:  # Move right
        col = min(col + 1, maze_width - 1)
    new_state = (row, col)
    return new_state

# Map action numbers to directions
def getChinesefromNum(action):
    action_dict = {0: "up", 1: "down", 2: "left", 3: "right"}
    return action_dict.get(action, "")

# Run the AI agent's optimal path
def run_maze(agent):
    agent.state = (init_position["x"], init_position["y"])  # Initialize agent's state to starting point
    screen.fill((0, 0, 0))  # Clear screen
    pygame.time.delay(500)

    while agent.state != (end_position["x"], end_position["y"]):  # End when the agent reaches the goal
        action = np.argmax(Q_table[agent.state[0], agent.state[1], :])  # Choose optimal action based on Q-values
        new_state = update_agent(agent, action)
        show_all()
        pygame.display.flip()  # Update display
        time.sleep(0.5)  # Delay for visualization
        agent.update_state(new_state)  # Update agent's state
    # Final display after reaching the goal
    show_all()
    time.sleep(0.5)

# Initialize Q-value table
Q_table = np.zeros((maze_height, maze_width, 4))  # 4 possible actions

# Q-Learning algorithm
def q_learning(agent, num_episodes, epsilon, learning_rate, discount_factor):
    global visualize
    for episode in range(num_episodes):
        agent.state = (init_position["x"], init_position["y"])  # Reset agent's state to starting point
        score = 0  # Track score
        steps = 0  # Track number of steps taken
        path = []  # Store path taken
        while agent.state != (end_position["x"], end_position["y"]):  # End when the agent reaches the goal
            action = agent.choose_action(epsilon)  # Choose action
            new_state = update_agent(agent, action)  # Update agent's position

            path.append(getChinesefromNum(action))  # Record the action taken

            # Reward structure based on the new state
            reward = -1 if maze[new_state] == 0 else -100  # Reward for moving to a path or penalty for hitting a wall

            # Update Q-value using the Q-learning formula
            Q_table[agent.state[0], agent.state[1], action] += learning_rate * \
                (reward + discount_factor * np.max(Q_table[new_state]) - Q_table[agent.state[0], agent.state[1], action])
            agent.update_state(new_state)  # Update agent's state
            score += reward  # Update score
            steps += 1  # Increment step count

        # Output current episode and optimal path length
        best_path_length = int(-score / 5)  # Calculate best path length
        if episode % 10 == 0:  # Print every 10 episodes
            print(f"Episode: {episode}, Path Length: {steps}")
            print(f"Movement Path: {path}")

# Pygame initialization
pygame.init()
cell_size = 40  # Size of each cell in the maze
screen = pygame.display.set_mode((maze_width * cell_size, maze_height * cell_size))  # Set up display
pygame.display.set_caption("Maze")  # Set window title

# Define the agent
agent = Agent((1, 1), [0, 1, 2, 3])  # Initialize agent at starting position with possible actions

# Run Q-learning algorithm
q_learning(agent, num_episodes, epsilon, learning_rate, discount_factor)

# Run the maze with the trained agent
run_maze(agent)

# Close Pygame
pygame.quit()