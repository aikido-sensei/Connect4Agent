import random
import numpy as np

class Agent:
    def __init__(self, delay_ms=500):
        self.delay_ms = delay_ms
    
    def get_valid_moves(self, board):
        """Returns a list of valid column moves."""
        valid_moves = []
        for col in range(7):
            if board[0][col] == 0:  # If top row is empty, column is valid
                valid_moves.append(col)
        return valid_moves
    
    def get_next_open_row(self, board, col):
        """Returns the next open row in the given column."""
        for r in range(5, -1, -1):
            if board[r][col] == 0:
                return r
        return -1
    
    def make_move(self, board):
        """Makes a random move on the board for now."""
        valid_moves = self.get_valid_moves(board)
        if valid_moves:
            return random.choice(valid_moves)
        return -1
    
    def get_delay(self):
        """Returns the delay in milliseconds before making a move."""
        return self.delay_ms 