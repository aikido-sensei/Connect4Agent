from math import sqrt

import numpy as np


class Node:
    """MCTS Node containing Q-values, prior probabilities, and visit counts"""

    def __init__(self, player: int, state: np.ndarray, move_count: int, prior=0):
        self.visit_count = 0
        self.player = player
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = state
        self.is_terminal = False
        self.terminal_value = None
        self.move_count = 0  # Track move count for value decay

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def ucb_score(self, child, puct):
        """Calculate UCB score using PUCT algorithm"""
        prior_score = puct * child.prior * sqrt(self.visit_count) / (1 + child.visit_count)

        if child.is_terminal and child.terminal_value == 0:  # Draw
            value_score = 0  # Draws are neutral
        else:
            value_score = child.value()
        return prior_score + value_score

    def get_best_child(self, puct):
        """Select child node using PUCT algorithm"""
        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action in self.children:
            child = self.children[action]
            score = self.ucb_score(child, puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    @staticmethod
    def backpropagate(search_path, value, total_moves, current_player, is_draw=False):
        """Backpropagate value through search path with temporal discounting"""
        discount_factor = 0.95
        for node in reversed(search_path):
            if node.player == current_player:
                mult = 1
            else:
                mult = -1

            # Apply temporal discount based on distance from end
            moves_to_end = total_moves - node.move_count
            discounted_value = mult * value * (discount_factor ** moves_to_end)

            if is_draw:
                # For draws, always add 0
                node.value_sum += 0
            else:
                # For wins/losses, use the discounted value
                node.value_sum += discounted_value

            node.visit_count += 1

    @staticmethod
    def choose_action(action_probs, temperature):

        if temperature == 0:
            action = np.argmax(action_probs)
        else:
            # Add small noise to probabilities for exploration
            noise = np.random.dirichlet([0.3] * len(action_probs))
            action_probs = 0.75 * action_probs + 0.25 * noise
            action_probs = action_probs / np.sum(action_probs)
            action = np.random.choice(7, p=action_probs)
        return action
