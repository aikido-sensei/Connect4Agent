import numpy as np
from enum import IntEnum


class PlayerTypes(IntEnum):
    PLAYER1 = 1
    PLAYER2 = 2
    
    
class PlayerTracker:
    def __init__(self):
        self.active_player = PlayerTypes.PLAYER1
    
    def is_p1_turn(self):
        return self.active_player == PlayerTypes.PLAYER1
    
    def switch_players(self):
        self.active_player = PlayerTypes.PLAYER2 if self.is_p1_turn() else PlayerTypes.PLAYER1

    def piece(self):
        return self.active_player.value


class Board:
    def __init__(self):
        """ Create a 6 x 7 board."""
        self.board = np.zeros((6, 7))
        
    def clear(self):
        """ Remove all pieces from the board."""
        self.board = np.zeros((6, 7))
    
    def get_next_open_row(self, col):
        """ On a selected column, find the row on which a piece can be added"""
        for r in range(5, -1, -1):
            if self.board[r][col] == 0:
                return r
        return -1

    def is_valid_location(self, col):
        """ Check whether the given column is full or not.
        Return True if the column is not full, False otherwise"""
        return self.board[0][col] == 0
    
    def get_valid_moves(self):
        """Returns list of valid moves"""
        return [col for col in range(7) if self.is_valid_location(col)]
    
    def drop_piece(self, row, col, piece):
        """ Adds the given piece at the given spot on the board.
        We assume that the row-column combination is a valid move"""
        self.board[row][col] = piece
    
    def has_won(self, piece):
        """ Determine whether a given player has won.
        To win, 4 pieces owned by the player need to be aligned vertically, horizontally, or diagonally."""
        # Check horizontal locations
        for c in range(4):
            for r in range(6):
                if self.board[r][c] == piece and self.board[r][c + 1] == piece and \
                        self.board[r][c + 2] == piece and self.board[r][c + 3] == piece:
                    return True
        
        # Check vertical locations
        for c in range(7):
            for r in range(3):
                if self.board[r][c] == piece and self.board[r + 1][c] == piece and \
                        self.board[r + 2][c] == piece and self.board[r + 3][c] == piece:
                    return True
        
        # Check positively sloped diagonals
        for c in range(4):
            for r in range(3):
                if self.board[r][c] == piece and self.board[r + 1][c + 1] == piece and \
                        self.board[r + 2][c + 2] == piece and self.board[r + 3][c + 3] == piece:
                    return True
        
        # Check negatively sloped diagonals
        for c in range(4):
            for r in range(3, 6):
                if self.board[r][c] == piece and self.board[r - 1][c + 1] == piece and \
                        self.board[r - 2][c + 2] == piece and self.board[r - 3][c + 3] == piece:
                    return True
        
        return False
    
    def has_draw(self):
        """Determine whether the game ended in a draw.
        A draw occurs when the entire board is full of pieces."""
        for col in range(7):
            if self.is_valid_location(col):
                return False
        return True
    
    def get_next_state(self, action, piece):
        """Returns next state after taking action"""
        next_board = Board()
        next_board.board = self.board.copy()
        row = next_board.get_next_open_row(action)
        if row != -1:
            next_board.drop_piece(row, action, piece)
        return next_board

    def value_at_slot(self, row, col):
        return self.board[row][col]
    
    
    
    
    
    