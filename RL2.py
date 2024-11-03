import numpy as np
import random
import pickle

class TicTacToe:
    def __init__(self):
        self.board = [' ' for _ in range(9)]  # Initialize empty board
        self.current_winner = None  # Track the winner!

    def print_board(self):
        # Print the board
        for row in [self.board[i*3:(i+1)*3] for i in range(3)]:
            print('| ' + ' | '.join(row) + ' |')

    def available_moves(self):
        return [i for i, x in enumerate(self.board) if x == ' ']

    def make_move(self, position, letter):
        if self.board[position] == ' ':
            self.board[position] = letter
            if self.winner(position, letter):
                self.current_winner = letter
            return True
        return False

    def winner(self, position, letter):
        # Winning conditions
        row_ind = position // 3
        row = self.board[row_ind*3:(row_ind+1)*3]
        if all([spot == letter for spot in row]):
            return True
        col_ind = position % 3
        column = [self.board[col_ind+i*3] for i in range(3)]
        if all([spot == letter for spot in column]):
            return True
        if position % 2 == 0:
            diagonal1 = [self.board[i] for i in [0, 4, 8]]
            if all([spot == letter for spot in diagonal1]):
                return True
            diagonal2 = [self.board[i] for i in [2, 4, 6]]
            if all([spot == letter for spot in diagonal2]):
                return True
        return False

    def is_full(self):
        return ' ' not in self.board

    def reset(self):
        self.board = [' ' for _ in range(9)]
        self.current_winner = None


class QLearningAgent:
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.q_table = {}  # Q-table
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def best_action(self, state, available_moves):
        q_values = [self.get_q_value(state, a) for a in available_moves]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(available_moves, q_values) if q == max_q]
        return random.choice(best_actions) if best_actions else random.choice(available_moves)

    def choose_action(self, state, available_moves):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_moves)
        else:
            return self.best_action(state, available_moves)

    def update_q_value(self, state, action, reward, next_state, available_moves):
        old_q = self.get_q_value(state, action)
        future_q = max([self.get_q_value(next_state, a) for a in available_moves], default=0)
        self.q_table[(state, action)] = old_q + self.alpha * (reward + self.gamma * future_q - old_q)

    def save(self, filename='qtable.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, filename='qtable.pkl'):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)


def get_state(board):
    return ''.join(board)


def train_agent(episodes=5000):
    game = TicTacToe()
    agent = QLearningAgent()
    for episode in range(episodes):
        game.reset()
        state = get_state(game.board)
        while not game.is_full():
            action = agent.choose_action(state, game.available_moves())
            game.make_move(action, 'X')
            if game.current_winner == 'X':
                agent.update_q_value(state, action, 1, None, [])
                break
            elif game.is_full():
                agent.update_q_value(state, action, 0, None, [])
                break

            opponent_action = random.choice(game.available_moves())
            game.make_move(opponent_action, 'O')
            if game.current_winner == 'O':
                agent.update_q_value(state, action, -1, None, [])
                break

            next_state = get_state(game.board)
            agent.update_q_value(state, action, 0, next_state, game.available_moves())
            state = next_state
    return agent


def play_game(agent):
    game = TicTacToe()
    game.print_board()

    while not game.is_full():
        user_action = int(input("Enter position (0-8): "))
        if game.make_move(user_action, 'O'):
            if game.current_winner == 'O':
                print("You win!")
                game.print_board()
                return
            elif game.is_full():
                print("It's a tie!")
                game.print_board()
                return

        state = get_state(game.board)
        agent_action = agent.choose_action(state, game.available_moves())
        game.make_move(agent_action, 'X')
        print("AI chose position", agent_action)
        game.print_board()

        if game.current_winner == 'X':
            print("AI wins!")
            return

    print("It's a tie!")


# Train the agent
print("Training the agent...")
agent = train_agent(episodes=5000)
print("Training completed.")

# Play the game
print("\nLet's play Tic-Tac-Toe!")
play_game(agent)
