from connect4_agent import Connect4Agent
import matplotlib.pyplot as plt
import os
from board import *
from monte_carlo import Node
from hyperparameters import Hyperparameters


def make_move(agent: Connect4Agent, board: np.ndarray, current_player: int, game_history: list,
              move_count: int, temperature: float):
    # Get current state
    state = board.copy()
    # Get action probabilities from MCTS
    action_probs = agent.get_action_probs(state, current_player, temperature)

    # Store state and probabilities
    game_history.append({
        'state': state.copy(),
        'current_player': current_player,
        'policy': action_probs,
        'move_number': move_count,
        'value': 0  # Initialize value to 0
    })

    # Select action
    action = Node.choose_action(action_probs, temperature)

    # Make move
    row = get_next_open_row(board, action)
    if row == -1:  # Invalid move
        # Penalize invalid moves heavily
        for history in game_history:
            history['value'] = -1.0 if history['current_player'] == current_player else 1.0
        return True

    drop_piece(board, row, action, current_player)

    # Check if game is over
    game_over = False
    value = 0

    if winning_move(board, current_player):
        game_over = True
        value = discount_value(1, move_count, agent.use_discounting)
    elif is_draw(board):
        game_over = True
        value = 0  # Draws are neutral

    if game_over:
        # Update values for all positions in the game
        for history in game_history:
            if value == 0:  # Draw
                history['value'] = 0
            elif history['current_player'] == current_player:
                history['value'] = value  # Winner gets positive value
            else:
                history['value'] = -value  # Loser gets negative of winner's value
    return game_over


def play_game(agent1, agent2, current_player, temperature):
    """Play a game between two agents"""
    board = create_board()
    game_history = []
    move_count = 0

    while True:
        move_count += 1
        current_agent = agent1 if current_player == 1 else agent2
        over = make_move(current_agent, board, current_player, game_history, move_count, temperature)
        if over:
            return game_history
        current_player = change_players(current_player)  # Switch players (1 -> 2 or 2 -> 1)


def train_agent(params1: Hyperparameters, params2: Hyperparameters):
    """Main training loop following AlphaGo Zero methodology"""
    iters = 100

    # Collect game stats
    episode_lengths = []
    wins = 0
    losses = 0
    draws = 0

    # Training iterations
    for iteration in range(iters):
        # initialize agents based on hyperparameters
        agent1 = params1.init_agent()
        params1.load_by_config(agent1)
        agent2 = params2.init_agent()
        params2.load_by_config(agent2)

        print(f"\n{'=' * 50}")
        print(f"Game {iteration + 1}/{iters}")
        print(f"{'=' * 50}")

        for player_start in [1, 2]:
            # play with temp set to 0, since we want pure exploitation
            game_history = play_game(agent1, agent2, player_start, temperature=0)

            episode_lengths.append(len(game_history))

            # Count game outcomes
            final_state = game_history[-1]
            if final_state['value'] == 0:  # Draw
                draws += 1
            elif final_state['value'] > 0:  # Win
                wins += 1
            else:  # Loss
                losses += 1

        # Calculate statistics
        total_games = wins + losses + draws
        win_rate = wins / total_games
        draw_rate = draws / total_games
        loss_rate = losses / total_games

        print(f"\nGame Statistics:")
        print(f"Average game length: {np.mean(episode_lengths):.1f} moves")
        print(f"Win rate: {win_rate * 100:.1f}%")
        print(f"Draw rate: {draw_rate * 100:.1f}%")
        print(f"Loss rate: {loss_rate * 100:.1f}%")



if __name__ == "__main__":
    # Start training with improved parameters
    train_agent(
        Hyperparameters(1),
        Hyperparameters(3)
    )
