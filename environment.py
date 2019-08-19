import json
import random
import numpy as np
from os import path

from tf_agents.trajectories import time_step
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec


from game import GameDataJsonDecoder, Player, ImpossibleBuildException
from enum import Enum


class Action(Enum):
    BUILD_STRUCTURE = 0
    BUILD_WONDER_STAGE = 1
    DISCARD = 2


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

        self._action_spec = [
            array_spec.BoundedArraySpec(
                shape=(), dtype=np.int32, minimum=0, maximum=2, name='action'),
            array_spec.BoundedArraySpec(
                shape=(), dtype=np.int32, minimum=0, maximum=7, name='card')
        ]
        self._observation_spec = {
            'age': array_spec.ArraySpec((), np.int32),
            'turn': array_spec.ArraySpec((), np.int32),
            'players_coins': array_spec.ArraySpec((self.player_count,), np.int32)
            #'player_decks': array_spec.ArraySpec((self.player_count,), np.int32),
            #'player_deck_offset': self.player_deck_offset,
            #'discarded_structures': self.discarded_structures
        }

    def to_observation(self):
        observation = {
            'age': np.int32(self.age),
            'turn': np.int32(self.turn),
            #'players_coins': [],
            # 'current_player': np.int32(self.current_player_index)
            # 'player_decks': self.player_decks,
            # 'player_deck_offset': self.player_deck_offset,
            # 'discarded_structures': self.discarded_structures
        }

        player_coins = []
        for player in self.players:
            player_coins.append(np.int32(player.coins))

        observation['players_coins'] = np.array(player_coins)
        return observation

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
        return time_step.restart(self.to_observation())

    def _step(self, player_actions):
        player_index = self.current_player_index

        player_action = Action(int(player_actions[0]))
        structure_index = int(player_actions[1])

        player = self.players[player_index]
        player_deck = self.player_deck(player_index)
        reward_malus = 0

        game_running = self.age <= 3
        if game_running:
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
                #reward_malus = -100

            self.finish_player_turn()

        observation = self.to_observation()
        # reward is only point from this turn
        reward = self.calculate_reward(player_index) + reward_malus

        if game_running:
            print("transition player " + str(self.current_player_index) + " action " + str(player_action.name)
                  + " turn " + str(self.turn) + " age " + str(self.age) + " reward " + str(reward))
            return time_step.transition(observation, reward)
        else:
            return time_step.termination(observation, reward)

    def finish_player_turn(self):
        self.current_player_index = (self.current_player_index + 1) % self.player_count
        if self.current_player_index == 0:
            self.finish_turn()
            if self.turn == 7:
                self.finish_age()

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
            structures = [s for s in json.load(structures_data_file) if s['minPlayerCount'] <= self.player_count]

        if self.age == 3:
            with open(path.join(path.dirname(__file__),
                                'game-data/guild-structures.json')) as guilds_data_file:
                guilds = json.load(guilds_data_file)

            random.shuffle(guilds)
            structures.extend(guilds[:self.player_count + 2])

        random.shuffle(structures)
        return [structures[i::self.player_count] for i in range(self.player_count)]

    def player_deck(self, player_index):
        if self.age > 3:
            return []

        return self.player_decks[(player_index + self.player_deck_offset) % self.player_count]

    def calculate_reward(self, player_index):
        new_player_score = self.players[player_index].score()
        reward = new_player_score - self.current_player_scores[player_index]
        self.current_player_scores[player_index] = new_player_score
        return reward

