import json
import random
import numpy as np
from os import path

from tensorflow.python.keras.utils.np_utils import to_categorical
from tf_agents.trajectories import time_step
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec

from game import GameDataJsonDecoder, Player, ImpossibleBuildException, Resource, Type
from enum import Enum


class Action(Enum):
    BUILD_STRUCTURE = 0
    BUILD_WONDER_STAGE = 1
    DISCARD = 2


player_hand_size = 7
# type (7), gold, resources (7), production (7), points, military, science (3)
card_observation_length = 7 + 1 + 7 + 7 + 1 + 1 + 3


class GameEnvironment(py_environment.PyEnvironment):

    def __init__(self, player_count=7):
        super().__init__()

        self.player_count = player_count
        self.current_player_index = 0

        self.age = 1
        self.turn = 1
        self.players = self.create_players()
        self.current_player_scores = [0 for _ in range(self.player_count)]
        self.player_decks = self.shuffle_age_structures()
        self.player_deck_offset = 0
        self.discarded_structures = []
        self._episode_ended = False

        # noinspection PyTypeChecker
        self._action_spec = [
            array_spec.BoundedArraySpec(
                shape=(), dtype=np.int32, minimum=0, maximum=len(Action) - 1, name='action'),
            array_spec.BoundedArraySpec(
                shape=(), dtype=np.int32, minimum=0, maximum=player_hand_size - 1, name='card')
        ]
        self._observation_spec = {
            'age': array_spec.ArraySpec((), np.float32),
            'turn': array_spec.ArraySpec((), np.float32),
            'players_coins': array_spec.ArraySpec((self.player_count,), np.float32),
            'player_hand': array_spec.ArraySpec((player_hand_size, card_observation_length), np.float32),
        }

    def to_observation(self):
        observation = {
            'age': np.float32(self.age),
            'turn': np.float32(self.turn),
            'players_coins': np.array(self.players_coins(), dtype=np.float32),
            # 'player_hand': []
        }

        player_hand = []
        for card in self.player_deck(self.current_player_index):
            player_hand.append(self.card_to_observation(card))
        while len(player_hand) < player_hand_size:
            player_hand.append(np.zeros(card_observation_length, dtype=np.float32))
        observation['player_hand'] = np.array(player_hand)

        return observation

    def players_coins(self):
        player_coins = []
        for player in self.players:
            player_coins.append(player.coins)
        return player_coins

    # noinspection PyTypeChecker
    @staticmethod
    def card_to_observation(card):
        # type (7), gold, resources (7), production (7), points, military, science (3)
        observation = []
        observation.extend(to_categorical(card['type'].value, num_classes=len(Type)))
        observation.append(card['cost']['gold'])
        for resource in Resource:
            if resource.name in card['cost']['resources']:
                observation.append(card['cost']['resources'][resource.name])
            else:
                observation.append(0)
        for resource in Resource:
            if resource.name in card['cost']['resources']:
                observation.append(card['cost']['resources'][resource.name])
            else:
                observation.append(0)
        return np.array(observation, dtype=np.float32)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.age = 1
        self.turn = 1
        self.players = self.create_players()
        self.player_decks = self.shuffle_age_structures()
        self.player_deck_offset = 0
        self.discarded_structures = []
        self._episode_ended = False
        return time_step.restart(self.to_observation())

    def _step(self, player_actions):
        if self._episode_ended:
            # print("game already ended resetting")
            return self.reset()

        player_index = self.current_player_index

        player_action = Action(int(player_actions[0]))
        structure_index = int(player_actions[1])

        player = self.players[player_index]
        player_deck = self.player_deck(player_index)

        success_reward_modifier = 0
        if structure_index >= len(player_deck):
            structure_index = len(player_deck) - 1

        structure = player_deck.pop(structure_index)
        try:
            if player_action == Action.BUILD_STRUCTURE:
                player.build_structure(structure)
            elif player_action == Action.BUILD_WONDER_STAGE:
                player.build_wonder_stage()
            elif player_action == Action.DISCARD:
                player.discard_structure()
            # print("Player " + str(player_index) + " choose to " + player_action.name + " " + structure['name'])
        except ImpossibleBuildException:
            player.discard_structure()
            success_reward_modifier = -3

        self.finish_player_turn()

        observation = self.to_observation()
        reward = self.calculate_score_difference(player_index) + success_reward_modifier

        if self._episode_ended:
            # print("game terminated at age " + str(self.age) + " reward " + str(reward))
            return time_step.termination(observation, reward)
        else:
            # print("transition player " + str(self.current_player_index) + " action " + str(player_action.name)
            #      + " turn " + str(self.turn) + " age " + str(self.age) + " reward " + str(reward))
            return time_step.transition(observation, reward)

    def finish_player_turn(self):
        self.current_player_index = (self.current_player_index + 1) % self.player_count
        if self.current_player_index == 0:
            self.finish_turn()

        if self.turn == 7:
            self.finish_age()

        if self.age == 4:
            self._episode_ended = True

    def finish_turn(self):
        self.turn += 1

    def finish_age(self):
        self.age += 1
        self.turn = 1

        if self.age <= 3:
            self.player_decks = self.shuffle_age_structures()

        if self.age == 2:
            self.player_deck_offset -= 1
        else:
            self.player_deck_offset += 1

        for player in self.players:
            player.resolve_military_conflicts(self.age)

    def create_players(self):
        with open(path.join(path.dirname(__file__), 'game-data/wonders.json')) as wonders_data_file:
            wonders = json.load(wonders_data_file, cls=GameDataJsonDecoder)
        random.shuffle(wonders)

        players = []
        for i in range(self.player_count):
            # TODO choose A or B
            player = Player(wonders[i]['sides']['A'])
            players.append(player)
        for i in range(len(players)):
            players[i].with_neighbor(players[i - 1], players[(i + 1) % self.player_count])

        return players

    def shuffle_age_structures(self):
        with open(path.join(path.dirname(__file__),
                            'game-data/age-' + str(self.age) + '-structures.json')) as structures_data_file:
            structures = [s for s in json.load(structures_data_file, cls=GameDataJsonDecoder) if s['minPlayerCount'] <= self.player_count]

        if self.age == 3:
            with open(path.join(path.dirname(__file__),
                                'game-data/guild-structures.json')) as guilds_data_file:
                guilds = json.load(guilds_data_file, cls=GameDataJsonDecoder)

            random.shuffle(guilds)
            structures.extend(guilds[:self.player_count + 2])

        random.shuffle(structures)
        return [structures[i::self.player_count] for i in range(self.player_count)]

    def player_deck(self, player_index):
        if self.age > 3:
            return []

        return self.player_decks[(player_index + self.player_deck_offset) % self.player_count]

    def calculate_score_difference(self, player_index):
        new_player_score = self.players[player_index].score()
        reward = new_player_score - self.current_player_scores[player_index]
        self.current_player_scores[player_index] = new_player_score
        return reward
