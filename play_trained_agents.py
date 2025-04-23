import argparse
import time
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
        print(f"WIN for player {current_player} in {move_count} moves.")
    elif is_draw(board):
        game_over = True
        print("DRAW")

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
        if args.progress:
            print(board, "\n")
            time.sleep(0.5)
        if over:
            return move_count
        current_player = change_players(current_player)  # Switch players (1 -> 2 or 2 -> 1)


def evaluate(params1: Hyperparameters, params2: Hyperparameters):
    """Main training loop following AlphaGo Zero methodology"""
    # Only one game is played since the agents will exploit their policy

    # initialize agents based on hyperparameters
    agent1 = params1.init_agent()
    params1.load_by_config(agent1)
    agent2 = params2.init_agent()
    params2.load_by_config(agent2)

    print("|||||||||||||\n\n")
    print("player 1 is config", params1.config)
    print("player 2 is config", params2.config)
    # play with temp set to 0, since we want pure exploitation
    move_count = play_game(agent1, agent2, 1, temperature=0)

    # reinitialize agent entirely, just in case
    agent1 = params1.init_agent()
    params1.load_by_config(agent1)
    agent2 = params2.init_agent()
    params2.load_by_config(agent2)

    print("|||||||||||||\n\n")
    print("player 1 is config", params2.config)
    print("player 2 is config", params1.config)
    # play with temp set to 0, since we want pure exploitation
    move_count = play_game(agent2, agent1, 1, temperature=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pit two agents against each other. The hyperparameters in each config are described in the paper.")
    parser.add_argument("--config1", type=int, choices=[0, 1, 2, 3], default=0, help="First config.")
    parser.add_argument("--config2", type=int, choices=[0, 1, 2, 3], default=0, help="Second config.")
    parser.add_argument("--progress", type=bool, default=False, help="Show board as game is played.")

    args = parser.parse_args()
    evaluate(
        Hyperparameters(args.config1),
        Hyperparameters(args.config2)
    )
