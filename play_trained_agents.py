import argparse
import time
from connect4_agent import Connect4Agent
import matplotlib.pyplot as plt
import os
from board import *
from monte_carlo import Node
from hyperparameters import Hyperparameters


def make_move(agent: Connect4Agent, board: np.ndarray, current_player: int, move_count: int, temperature: float):
    # Get current state
    state = board.copy()
    # Get action probabilities from MCTS
    action_probs = agent.get_action_probs(state, current_player, temperature)

    # Select action
    action = Node.choose_action(action_probs, temperature)

    # Make move
    row = get_next_open_row(board, action)

    drop_piece(board, row, action, current_player)

    # Check if game is over
    game_over = False
    value = 0

    if winning_move(board, current_player):
        game_over = True
        print(f"WIN for player {current_player} in {move_count} moves.")
        value = current_player
    elif is_draw(board):
        game_over = True
        print("DRAW")

    return game_over, value


def play_game(agent1, agent2, current_player, temperature):
    """Play a game between two agents"""
    board = create_board()
    move_count = 0

    while True:
        move_count += 1
        current_agent = agent1 if current_player == 1 else agent2
        over, value = make_move(current_agent, board, current_player, move_count, temperature)
        if args.progress:
            print(board, "\n")
            time.sleep(0.5)
        if over:
            return move_count, value
        current_player = change_players(current_player)  # Switch players (1 -> 2 or 2 -> 1)


def do_state_update(game_stats: dict, current_winner: int, non_random_agent: int, move_count: int):
    if current_winner == -1:
        return game_stats

    if current_winner == 0:
        game_stats["draws"] += 1
    elif current_winner == non_random_agent:
        game_stats["wins"] += 1
    else:
        game_stats["losses"] += 1

    game_stats["move_count"][non_random_agent - 1] += move_count
    return game_stats


def evaluate(params1: Hyperparameters, params2: Hyperparameters):
    """Main training loop following AlphaGo Zero methodology"""
    if params1.config == -1:
        # Play many games against random agent to get an average
        num_games = 200
    else:
        # For trained agents, only one game is played since the agents will exploit their policy
        num_games = 1

    # draws, losses, wins
    game_stats = {
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "move_count": [0, 0]
    }

    # initialize agents based on hyperparameters
    for i in range(num_games):
        print(f"\n{'=' * 50}")
        agent1 = params1.init_agent()
        params1.load_by_config(agent1)
        agent2 = params2.init_agent()
        params2.load_by_config(agent2)

        print("\nplayer 1 is config", params1.config)
        print("player 2 is config", params2.config)
        # play with temp set to 0, since we want pure exploitation
        move_count, winning_p = play_game(agent1, agent2, 1, temperature=0)

        if params1.config == -1 and winning_p != -1:
            game_stats = do_state_update(game_stats, winning_p, 2, move_count)

        # reinitialize agent entirely, just in case
        agent1 = params1.init_agent()
        params1.load_by_config(agent1)
        agent2 = params2.init_agent()
        params2.load_by_config(agent2)

        print("player 1 is config", params2.config)
        print("player 2 is config", params1.config)
        # play with temp set to 0, since we want pure exploitation
        move_count, winning_p = play_game(agent2, agent1, 1, temperature=0)

        if params1.config == -1 and winning_p != -1:
            game_stats = do_state_update(game_stats, winning_p, 1, move_count)
        print(f"\n{'=' * 50}")

    if params1.config == -1:
        print(f"wins: {game_stats['wins']},"
              f"\n draws: {game_stats['draws']}",
              f"\n losses: {game_stats['losses']}",
              f"\n move count as p1: {game_stats['move_count'][0] / num_games}",
              f"\n move count as p2: {game_stats['move_count'][1] / num_games}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pit two agents against each other. The hyperparameters in each config are described in the paper.")
    parser.add_argument("--config1", type=int, choices=[-1, 0, 1, 2, 3], default=0, help="First config.")
    parser.add_argument("--config2", type=int, choices=[0, 1, 2, 3], default=1, help="Second config.")
    parser.add_argument("--progress", type=bool, default=False, help="Show board as game is played.")

    args = parser.parse_args()
    evaluate(
        Hyperparameters(args.config1),
        Hyperparameters(args.config2)
    )
