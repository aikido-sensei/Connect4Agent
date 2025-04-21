import numpy as np


def create_board():
    return np.zeros((6, 7))


def drop_piece(board, row, col, piece):
    board[row][col] = piece


def is_valid_location(board, col):
    return board[0][col] == 0


def get_valid_moves(board):
    """Returns list of valid moves"""
    return [col for col in range(7) if board[0][col] == 0]


def get_next_open_row(board, col):
    for r in range(5, -1, -1):
        if board[r][col] == 0:
            return r
    return -1


def get_next_state(board: np.ndarray, action, player):
    """Returns next state after taking action"""
    next_board = board.copy()
    for row in range(5, -1, -1):
        if next_board[row][action] == 0:
            next_board[row][action] = player
            break
    return next_board


def winning_move(board, piece):
    # Check horizontal locations
    for c in range(4):
        for r in range(6):
            if board[r][c] == piece and board[r][c + 1] == piece and \
                    board[r][c + 2] == piece and board[r][c + 3] == piece:
                return True

    # Check vertical locations
    for c in range(7):
        for r in range(3):
            if board[r][c] == piece and board[r + 1][c] == piece and \
                    board[r + 2][c] == piece and board[r + 3][c] == piece:
                return True

    # Check positively sloped diagonals
    for c in range(4):
        for r in range(3):
            if board[r][c] == piece and board[r + 1][c + 1] == piece and \
                    board[r + 2][c + 2] == piece and board[r + 3][c + 3] == piece:
                return True

    # Check negatively sloped diagonals
    for c in range(4):
        for r in range(3, 6):
            if board[r][c] == piece and board[r - 1][c + 1] == piece and \
                    board[r - 2][c + 2] == piece and board[r - 3][c + 3] == piece:
                return True

    return False


def is_draw(board):
    return len([col for col in range(7) if board[0][col] == 0]) == 0


def change_players(current_player):
    return 3 - current_player
