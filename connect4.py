import pygame
import sys
from connect4_agent import Connect4Agent
from board import *
from hyperparameters import Hyperparameters

# Initialize Pygame
pygame.init()

# Constants
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)

# Game dimensions
SQUARESIZE = 100
RADIUS = int(SQUARESIZE / 2 - 5)
width = 7 * SQUARESIZE
height = 7 * SQUARESIZE
size = (width, height)

# Initialize fonts
title_font = pygame.font.SysFont("monospace", 75)
menu_font = pygame.font.SysFont("monospace", 50)
game_font = pygame.font.SysFont("monospace", 75)

# Game states
MENU = 0
PLAYING = 1
GAME_OVER = 2


def draw_board(board):
    # Draw the board background
    for c in range(7):
        for r in range(6):
            pygame.draw.rect(screen, BLUE, (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK, (
            int(c * SQUARESIZE + SQUARESIZE / 2), int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)), RADIUS)

    # Draw the pieces
    for c in range(7):
        for r in range(6):
            if board[r][c] == 1:
                pygame.draw.circle(screen, RED, (
                int(c * SQUARESIZE + SQUARESIZE / 2), int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)), RADIUS)
            elif board[r][c] == 2:
                pygame.draw.circle(screen, YELLOW, (
                int(c * SQUARESIZE + SQUARESIZE / 2), int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)), RADIUS)
    pygame.display.update()


def draw_menu():
    screen.fill(BLACK)

    # Draw title
    title = title_font.render("CONNECT 4", 1, WHITE)
    screen.blit(title, (width // 2 - title.get_width() // 2, 50))

    # Draw buttons
    human_button = pygame.Rect(width // 2 - 150, 200, 300, 80)
    agent_button = pygame.Rect(width // 2 - 150, 350, 300, 80)

    pygame.draw.rect(screen, GRAY, human_button)
    pygame.draw.rect(screen, GRAY, agent_button)

    human_text = menu_font.render("VS HUMAN", 1, WHITE)
    agent_text = menu_font.render("VS AGENT", 1, WHITE)

    screen.blit(human_text, (width // 2 - human_text.get_width() // 2, 220))
    screen.blit(agent_text, (width // 2 - agent_text.get_width() // 2, 370))

    pygame.display.update()
    return human_button, agent_button


def draw_game_over(winner, board):
    # Draw the final board state
    draw_board(board)

    # Create a semi-transparent overlay
    overlay = pygame.Surface((width, height - SQUARESIZE), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 128))
    screen.blit(overlay, (0, SQUARESIZE))

    # Draw winner message
    if winner == 0:
        message = "It's a tie!"
    else:
        message = f"Player {winner} wins!"
    win_text = game_font.render(message, 1, RED if winner == 1 else YELLOW)
    screen.blit(win_text, (width // 2 - win_text.get_width() // 2, SQUARESIZE + 50))

    # Draw menu button
    menu_button = pygame.Rect(width // 2 - 150, 20, 300, 60)
    pygame.draw.rect(screen, GRAY, menu_button)
    menu_text = pygame.font.SysFont("monospace", 40).render("BACK TO MENU", 1, WHITE)
    screen.blit(menu_text, (width // 2 - menu_text.get_width() // 2, 35))

    pygame.display.update()
    return menu_button


# Set up the display
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Connect 4")

# Load the trained agent
params = Hyperparameters(0)
agent = params.init_agent()
params.load_by_config(agent)
# try:
#     agent.load_model('models/model_latest.pth')
#     print("Loaded trained model successfully!")
# except:
#     print("No trained model found, using untrained model")

# Game state variables
game_state = MENU
board = create_board()
game_over = False
turn = 0
winner = 0
vs_agent = False

# Main game loop
while True:
    if game_state == MENU:
        screen.fill(BLACK)
        human_button, agent_button = draw_menu()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if human_button.collidepoint(pos):
                    game_state = PLAYING
                    vs_agent = False
                    board = create_board()
                    game_over = False
                    turn = 0
                    screen.fill(BLACK)
                    draw_board(board)
                elif agent_button.collidepoint(pos):
                    game_state = PLAYING
                    vs_agent = True
                    board = create_board()
                    game_over = False
                    turn = 0
                    screen.fill(BLACK)
                    draw_board(board)

    elif game_state == PLAYING:
        # Handle agent's turn if playing against agent
        if vs_agent and turn == 0:
            pygame.time.wait(500)  # Fixed 500ms delay for better UX

            # Get agent's move using MCTS
            action_probs = agent.get_action_probs(board, 1, temperature=0.5)
            col = np.argmax(action_probs)

            # Ensure move is valid
            if not is_valid_location(board, col):
                valid_moves = [c for c in range(7) if is_valid_location(board, c)]
                if valid_moves:
                    col = valid_moves[0]
                else:
                    continue

            row = get_next_open_row(board, col)
            drop_piece(board, row, col, 1)

            # Check for win
            if winning_move(board, 1):
                winner = 1
                game_state = GAME_OVER
            elif len([col for col in range(7) if is_valid_location(board, col)]) == 0:
                winner = 0
                game_state = GAME_OVER

            if game_state == PLAYING:
                draw_board(board)
                turn += 1
                turn = turn % 2
            else:
                menu_button = draw_game_over(winner, board)
            continue

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEMOTION:
                pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                posx = event.pos[0]
                if turn == 0:
                    pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE / 2)), RADIUS)
                else:
                    pygame.draw.circle(screen, YELLOW, (posx, int(SQUARESIZE / 2)), RADIUS)
            pygame.display.update()

            if event.type == pygame.MOUSEBUTTONDOWN:
                pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))

                posx = event.pos[0]
                col = int(posx / SQUARESIZE)

                if col < 0 or col > 6:
                    continue

                row = get_next_open_row(board, col)
                if row == -1:
                    continue

                piece = 1 if turn == 0 else 2
                drop_piece(board, row, col, piece)

                if winning_move(board, piece):
                    winner = piece
                    game_state = GAME_OVER
                elif len([col for col in range(7) if is_valid_location(board, col)]) == 0:
                    winner = 0
                    game_state = GAME_OVER

                if game_state == PLAYING:
                    draw_board(board)
                    turn += 1
                    turn = turn % 2
                else:
                    menu_button = draw_game_over(winner, board)

    elif game_state == GAME_OVER:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if menu_button.collidepoint(pos):
                    game_state = MENU
