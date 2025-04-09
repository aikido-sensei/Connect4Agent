import pygame
import sys
import numpy as np
from connect4_agent import Connect4Agent
from board import Board, PlayerTracker

# Initialize Pygame
pygame.init()

# Game dimensions
SQUARESIZE = 100
RADIUS = int(SQUARESIZE / 2 - 5)
width = 7 * SQUARESIZE
height = 7 * SQUARESIZE
size = (width, height)

# Game states
MENU = 0
PLAYING = 1
GAME_OVER = 2

# Colours
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

player_colours = {0: BLACK, 1: RED, 2: YELLOW}  # Map from player number (0 = empty) to colour

# Initialize fonts
title_font = pygame.font.SysFont("monospace", 75)
menu_font = pygame.font.SysFont("monospace", 50)
game_font = pygame.font.SysFont("monospace", 75)


def draw_board(screen, board: Board):
    for c in range(7):
        for r in range(6):
            # Draw the board background
            pygame.draw.rect(screen, BLUE, (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
            
            # Draw the pieces
            piece = board.value_at_slot(r, c)
            pygame.draw.circle(screen, player_colours[piece], (
                int(c * SQUARESIZE + SQUARESIZE / 2),
                int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)), RADIUS)
    
    pygame.display.update()


def draw_menu(screen):
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


def draw_game_over(screen, winner, board):
    # Draw the final board state
    draw_board(screen, board)
    
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


def main():
    # Set up the display
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Connect 4")
    
    # Load the trained agent
    agent = Connect4Agent(num_simulations=100)
    try:
        agent.load_model('models/model_latest.pth')
        print("Loaded trained model successfully!")
    except FileNotFoundError:
        print("No trained model found, using untrained model")
    
    # Game state variables
    game_state = MENU
    board = Board()
    player = PlayerTracker()
    winner = 0
    vs_agent = False
    menu_button = None
    
    # Main game loop
    while True:
        
        # Game type selection
        if game_state == MENU:
            screen.fill(BLACK)
            human_button, agent_button = draw_menu(screen)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    human_game = human_button.collidepoint(pos)
                    agent_game = agent_button.collidepoint(pos)
                    
                    # Since buttons do not overlap, only one of the two conditions will be true
                    if human_game or agent_game:
                        game_state = PLAYING
                        player = PlayerTracker()
                        screen.fill(BLACK)
                        board.clear()
                        draw_board(screen, board)
                        vs_agent = agent_game
        
        # Actual game loop
        elif game_state == PLAYING:
            # Handle agent's turn if playing against agent
            if vs_agent and player.is_p1_turn():
                pygame.time.wait(500)  # Fixed 500ms delay for better UX
                
                # Get agent's move using MCTS
                action_probs = agent.get_action_probs(board, player.piece(), temperature=0.5)
                col = np.argmax(action_probs)
                
                # Ensure move is valid
                if not board.is_valid_location(col):
                    valid_moves = [c for c in range(7) if board.is_valid_location(c)]
                    if valid_moves:
                        col = valid_moves[0]
                    else:
                        continue
                
                row = board.get_next_open_row(col)
                board.drop_piece(row, col, player.piece())
            
            # Handle human's turn
            else:
                made_choice = False
                while not made_choice:
                    # Show where piece will fall
                    posx = pygame.mouse.get_pos()[0]
                    pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                    pygame.draw.circle(screen, player_colours[player.piece()], (posx, int(SQUARESIZE / 2)), RADIUS)
                    pygame.display.update()
                    
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                        
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                            
                            posx = event.pos[0]
                            col = int(posx / SQUARESIZE)
                            
                            if col < 0 or col > 6:
                                continue
                            
                            row = board.get_next_open_row(col)
                            if row == -1:
                                continue
                            
                            board.drop_piece(row, col, player.piece())
                            made_choice = True
            
            # Check for win
            if board.has_won(player.piece()):
                winner = player.piece()
                game_state = GAME_OVER
            elif board.has_draw():
                winner = 0
                game_state = GAME_OVER
            
            draw_board(screen, board)
            if game_state == PLAYING:
                player.switch_players()
            else:
                menu_button = draw_game_over(screen, winner, board)
        
        # End screen showing who won
        elif game_state == GAME_OVER:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    if menu_button.collidepoint(pos):
                        game_state = MENU


if __name__ == "__main__":
    main()
