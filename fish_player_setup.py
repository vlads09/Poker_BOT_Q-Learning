'''useful imports'''
import pickle
import random

import numpy as np
from pypokerengine.players import BasePokerPlayer

class FishPlayer(BasePokerPlayer):

    def __init__(self):
        super().__init__()
        self.q_table = {}  # dictionary of q-table to store q-values
        self.epsilon = 1  # exploration rate
        self.alpha = 0.7  # learning rate
        self.gamma = 0.95  # discounting rate
        self.hole_card = [] # cards owned by the FishPlayer
        self.valid_actions = [] # available actions: fold, call, raise
        self.last_state = None # update the q-table
        self.last_action_idx = -1 # update action made
        self.load_q_table("progress.pkl") # loading the progress made in other games

    def save_q_table(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename):
        try:
            with open(filename, "rb") as f:
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            self.q_table = {}

    def get_state(self, hole_card, round_state):
        # state representation in q-table
        return str(hole_card) + " " + str(round_state['street'])

    def get_reward(self, winners):
        # reward functionality: simplistic reward based on the outcome of the round
        for winner in winners:
            if winner['uuid'] == self.uuid:
                return 1  # won the round
        return -1  # lost the round

    def is_pair(self, hole_card):
        return hole_card[0][1] == hole_card[1][1]

    def is_high_cards(self, hole_card):
        rank_dict = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                     '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        return rank_dict[hole_card[0][1]] > 8 and rank_dict[hole_card[1][1]] > 8

    def declare_action(self, valid_actions, hole_card, round_state):
        self.valid_actions = valid_actions
        self.hole_card = hole_card

        state = self.get_state(hole_card, round_state)

        # if there is a new state, create one
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(valid_actions))

        # check for pair and high cards in FishPlayer's hand
        has_pair = self.is_pair(hole_card)
        has_high_cards = self.is_high_cards(hole_card)
        if round_state['street'] == 'preflop':
            if has_pair:
                # bias towards raising or calling when having a pair
                if random.uniform(0, 1) < 0.7:  # 70% chance to raise
                    action_idx = 2
                else:
                    action_idx = 1
            elif has_high_cards:
                # bias towards calling when having high cards
                action_idx = 1
            else:
                # if number less than epsilon -> random choice
                # else: exploitation (taking the biggest Q value for this state)
                if random.uniform(0, 1) < self.epsilon:
                    action_idx = random.choice(range(len(valid_actions)))
                else:
                    action_idx = np.argmax(self.q_table[state])
        else:
            if random.uniform(0, 1) < self.epsilon:
                action_idx = random.choice(range(len(valid_actions)))
            else:
                action_idx = np.argmax(self.q_table[state])

        action, amount = valid_actions[action_idx]["action"], valid_actions[action_idx]["amount"]

        if action == "raise":
            action_info = valid_actions[2]
            amount = action_info["amount"]["min"]
            if amount == -1:
                action = "call"
                action_info = valid_actions[1]
                amount = action_info["amount"]
        elif action == "call":
            action_info = valid_actions[1]
            amount = action_info["amount"]
        elif action == "fold":
            action_info = valid_actions[0]
            amount = action_info["amount"]

        self.last_state = state
        self.last_action_idx = action_idx

        return action, amount

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        # calculate reward
        reward = self.get_reward(winners)
        print(f'reward is {reward}')
        new_state = self.get_state(self.hole_card, round_state)
        if new_state not in self.q_table:
            self.q_table[new_state] = np.zeros(len(self.valid_actions))

        # if there is no entry, it means it is the beginning of the learning process
        # and the round ended shortly
        if self.last_state in self.q_table.keys():
            # retrieve the old q-value
            old_value = self.q_table[self.last_state][self.last_action_idx]
            # take the highest q-value among all possible actions in the new state
            next_max = np.max(self.q_table[new_state])
            # q-learning update rule
            new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
            print(f'value for q_table is {new_value}')
            # update the q-table
            self.q_table[self.last_state][self.last_action_idx] = new_value
        print(self.q_table)
        print()
        # decay epsilon to reduce exploration over time in order to make the FishPlayer use
        # what they learnt
        self.epsilon = max(0.4, self.epsilon * 0.995)
        # save progress
        self.save_q_table("progress.pkl")

def setup_ai():
    return FishPlayer()
