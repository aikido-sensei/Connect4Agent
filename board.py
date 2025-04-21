import numpy as np


def create_board():
    """ Make a new connect 4 board."""
    return np.zeros((6, 7))


def drop_piece(board, row, col, piece):
    """ Add a piece at the desired row and column. We assume that the location is valid
    and respects the rules of the game."""
    board[row][col] = piece


def is_valid_location(board, col):
    """ Check whether the current column can accept pieces (True if not full, false otherwise)."""
    return board[0][col] == 0


def get_valid_moves(board):
    """Returns list of valid moves."""
    return [col for col in range(7) if is_valid_location(board, col)]


def get_next_open_row(board, col):
    """ Get the row number that will allow a piece to be added to this column."""
    for r in range(5, -1, -1):
        if board[r][col] == 0:
            return r
    return -1


def get_next_state(board: np.ndarray, action, player):
    """Returns next state after taking action."""
    next_board = board.copy()
    for row in range(5, -1, -1):
        if next_board[row][action] == 0:
            next_board[row][action] = player
            break
    return next_board


def winning_move(board, piece):
    """ Check whether the desired piece type has won the game."""
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
    """ Check if the game ended in a draw (aka the board is full)."""
    return len([col for col in range(7) if board[0][col] == 0]) == 0


def change_players(current_player):
    """ Switch to the other player."""
    return 3 - current_player


def discount_value(value, move_count):
    """Discount the current win value."""
    if value == 0:
        return 0
    # Winning quickly is better (max reward 1.0 for quick win, min 0.3 for slow win)
    progress = move_count / 42  # How far into the game are we
    discounted = value - progress * 0.7  # Decrease reward for longer games
    return discounted
