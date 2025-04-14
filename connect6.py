import sys
import numpy as np
import random
import math
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime


class UCTNode:
    def __init__(self, state, turn, parent=None, action=None):
        self.state = state
        self.turn = turn
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = self.get_legal_actions()

    def get_legal_actions(self): #all the places we can put it !
        return [(r, c) for r in range(self.state.shape[0]) for c in range(self.state.shape[1]) if self.state[r, c] == 0]

    def fully_expanded(self): #if it is fully expanded (returns a boolean)
        return len(self.untried_actions) == 0

class UCTMCTS:
    def __init__(self, iterations=150000, exploration_constant=1.41):
        self.iterations = iterations
        self.c = exploration_constant

    def create_env_from_state(self, board, turn):
        env = Connect6Game(size=board.shape[0])
        env.board = board.copy()
        env.turn = turn
        return env

    def select_child(self, node):
        best_uct = -math.inf
        best_child = None

        for child in node.children.values():
            if child.visits == 0:
                uct = math.inf
            else:
                exploitation = child.total_reward / child.visits
                exploration = self.c * math.sqrt(math.log(node.visits + 1) / child.visits)
                uct = exploitation + exploration

            if uct > best_uct:
                best_uct = uct
                best_child = child

        return best_child

    def rollout(self, env, rollout_depth=7):
        action_count = 0
        max_moves = env.size * env.size #size of the board
        starting_turn = env.turn

        while not env.game_over and action_count < rollout_depth:
            legal_moves = [(r, c) for r in range(env.size) for c in range(env.size) if env.board[r, c] == 0]
            if not legal_moves:
                break
            action = random.choice(legal_moves)
            action_str = f"{chr(ord('A') + action[1] + (1 if action[1] >= 8 else 0))}{action[0]+1}"
            env.play_move('B' if env.turn == 1 else 'W', action_str, printing=False)
            winner = env.check_win()
            if winner:
                #print('is this happening?')
                if winner == starting_turn:
                    return (max_moves - action_count) / max_moves 
                else:
                    return 0.0
            action_count += 1
            if action_count > max_moves:
                break
        return 0

    def backpropagate(self, node, reward):
        while node:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def run_simulation(self, root):
        node = root
        env = self.create_env_from_state(root.state, root.turn)

        while node.fully_expanded() and node.children:
            node = self.select_child(node)
            action_str = f"{chr(ord('A') + node.action[1] + (1 if node.action[1] >= 8 else 0))}{node.action[0]+1}"
            env.play_move('B' if env.turn == 1 else 'W', action_str, printing=False)

        while node.untried_actions:
            action = random.choice(node.untried_actions)
            node.untried_actions.remove(action)
            if env.board[action[0], action[1]] != 0:
                continue
            
            action_str = f"{chr(ord('A') + action[1] + (1 if action[1] >= 8 else 0))}{action[0]+1}"
            env.play_move('B' if env.turn == 1 else 'W', action_str, printing=False)
            new_node = UCTNode(env.board.copy(), env.turn, parent=node, action=action)
            node.children[action] = new_node
            node = new_node
            break
        
        reward = self.rollout(env)
        self.backpropagate(node, reward)

    def best_action(self, root):
        return max(root.children.items(), key=lambda item: item[1].visits)[0]


class Connect6Game:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0: Empty, 1: Black, 2: White
        self.turn = 1  # 1: Black, 2: White
        self.game_over = False

    def reset_board(self):
        """Clears the board and resets the game."""
        self.board.fill(0)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)
    def set_board_size(self, size):
        """Sets the board size and resets the game."""
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)
    def check_win(self):
        """Checks if a player has won.
        Returns:
        0 - No winner yet
        1 - Black wins
        2 - White wins
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    current_color = self.board[r, c]
                    for dr, dc in directions:
                        prev_r, prev_c = r - dr, c - dc
                        if 0 <= prev_r < self.size and 0 <= prev_c < self.size and self.board[prev_r, prev_c] == current_color:
                            continue
                        count = 0
                        rr, cc = r, c
                        while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr, cc] == current_color:
                            count += 1
                            rr += dr
                            cc += dc
                        if count >= 6:
                            return current_color
        return 0

    def index_to_label(self, col):
        """Converts column index to letter (skipping 'I')."""
        return chr(ord('A') + col + (1 if col >= 8 else 0))  # Skips 'I'

    def label_to_index(self, col_char):
        """Converts letter to column index (accounting for missing 'I')."""
        col_char = col_char.upper()
        if col_char >= 'J':  # 'I' is skipped
            return ord(col_char) - ord('A') - 1
        else:
            return ord(col_char) - ord('A')

    def play_move(self, color, move):
        """Places stones and checks the game status."""
        if self.game_over:
            print("? Game over")
            return

        stones = move.split(',')
        positions = []

        for stone in stones:
            stone = stone.strip()
            if len(stone) < 2:
                print("? Invalid format")
                return
            col_char = stone[0].upper()
            if not col_char.isalpha():
                print("? Invalid format")
                return
            col = self.label_to_index(col_char)
            try:
                row = int(stone[1:]) - 1
            except ValueError:
                print("? Invalid format")
                return
            if not (0 <= row < self.size and 0 <= col < self.size):
                print("? Move out of board range")
                return
            if self.board[row, col] != 0:
                print("? Position already occupied")
                return
            positions.append((row, col))

        for row, col in positions:
            self.board[row, col] = 1 if color.upper() == 'B' else 2

        self.turn = 3 - self.turn
        print('= ', end='', flush=True)



    def generate_move(self, color):
        """Generates a move for the computer based on my logic!"""
        root = UCTNode(self.board.copy(), self.turn)
        iterations = 4000
        mcts = UCTMCTS(iterations=iterations)
            
        for _ in range(mcts.iterations):
            mcts.run_simulation(root)
            
        best_move = mcts.best_action(root)
        # Convert best_move (a tuple (row, col)) to the move string.
        move_str = f"{chr(ord('A') + best_move[1] + (1 if best_move[1] >= 8 else 0))}{best_move[0] + 1}"
        self.play_move(color, move_str)
        print(f"{move_str}\n\n", end='', flush=True)
        print(move_str, file=sys.stderr)
        return


    def show_board(self):
        """Displays the board as text."""
        print("= ")
        for row in range(self.size - 1, -1, -1):
            line = f"{row+1:2} " + " ".join("X" if self.board[row, col] == 1 else "O" if self.board[row, col] == 2 else "." for col in range(self.size))
            print(line)
        col_labels = "   " + " ".join(self.index_to_label(i) for i in range(self.size))
        print(col_labels)
        print(flush=True)

    def list_commands(self):
        """Lists all available commands."""
        print("= ", flush=True)  

    def process_command(self, command):
        """Parses and executes GTP commands."""
        command = command.strip()
        if command == "get_conf_str env_board_size:":
            print("env_board_size=19", flush=True)

        if not command:
            return
        
        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "boardsize":
            try:
                size = int(parts[1])
                self.set_board_size(size)
            except ValueError:
                print("? Invalid board size")
        elif cmd == "clear_board":
            self.reset_board()
        elif cmd == "play":
            if len(parts) < 3:
                print("? Invalid play command format")
            else:
                self.play_move(parts[1], parts[2])
                print('', flush=True)
        elif cmd == "genmove":
            if len(parts) < 2:
                print("? Invalid genmove command format")
            else:
                self.generate_move(parts[1])
        elif cmd == "showboard":
            self.show_board()
        elif cmd == "list_commands":
            self.list_commands()
        elif cmd == "quit":
            print("= ", flush=True)
            sys.exit(0)
        else:
            print("? Unsupported command")

    def run(self):
        """Main loop that reads GTP commands from standard input."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                self.process_command(line)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"? Error: {str(e)}")

if __name__ == "__main__":
    game = Connect6Game()
    game.run()
