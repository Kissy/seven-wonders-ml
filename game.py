import json
import numpy as np
from collections import defaultdict
from math import floor
from enum import Enum


class Type(Enum):
    RAW_MATERIAL = 1
    MANUFACTURED_GOOD = 2
    CIVILIAN = 3
    SCIENTIFIC = 4
    COMMERCIAL = 5
    MILITARY = 6
    GUILD = 7


class Resource(Enum):
    WOOD = 1
    STONE = 2
    ORE = 3
    CLAY = 4

    PAPYRUS = 5
    LOOM = 6
    GLASS = 7


class Science(Enum):
    WHEEL = 1
    COMPASS = 2
    TABLET = 3
    ANY = 4


class ImpossibleBuildException(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class Player:

    def __init__(self, wonder):
        self.wonder = wonder
        self.coins = 3
        self.neighbors = {'SELF': {
            'player': self,
            'commerce': defaultdict(lambda: 0)
        }}

        # Construction & Production
        self.wonder_stage = 0
        self.constructions = []
        self.productions = [self.wonder['production']]
        self.resources_for_sale = defaultdict(int, self.wonder['production'])
        self.free_build_available = False
        self.free_build_used_this_age = False

        # Score calculation
        self.shields = 0
        self.defeat_tokens = 0
        self.victory_points = 0
        self.wonder_points = 0
        self.civilian_points = 0
        self.scientific_symbols = defaultdict(int)
        self.copy_guild = False

    def __repr__(self) -> str:
        return str({
            'coins': np.int32(self.coins)
        })

    def with_neighbor(self, left, right):
        self.neighbors['LEFT'] = {
            'player': left,
            'commerce': defaultdict(lambda: 2)
        }
        self.neighbors['RIGHT'] = {
            'player': right,
            'commerce': defaultdict(lambda: 2)
        }
        pass

    def build_structure(self, structure):
        cost = self.build_cost(structure['cost'])
        if cost is None or self.coins < cost:
            raise ImpossibleBuildException('cannot build structure')

        self.coins -= cost
        self.constructions.append(structure)
        self.apply_effect(structure['effect'], structure['type'])

    def build_wonder_stage(self):
        if self.wonder_stage >= len(self.wonder['stages']):
            raise ImpossibleBuildException('no more wonder stage to build')

        cost = self.build_cost(self.wonder['stages'][self.wonder_stage]['cost'])
        if cost is None or self.coins < cost:
            raise ImpossibleBuildException('cannot pay for the wonder stage cost')

        self.coins -= cost
        self.wonder_stage += 1
        for effect in self.wonder['stages'][self.wonder_stage - 1]['effects']:
            self.apply_effect(effect, None)

    def discard_structure(self):
        self.coins += 3

    def build_cost(self, cost):
        # TODO self.free_build_available

        for construction in self.constructions:
            if 'structure' in cost and cost['structure'] == construction['name']:
                return 0

        resources = defaultdict(int, cost['resources'])

        if any(resources.values()):
            for production in self.productions:
                for resource, quantity in production.items():
                    if resources[resource] > 0:
                        resources[resource] -= quantity
                        break

        gold = 0
        gold += cost['gold']

        # TODO priorize
        if any(resources.values()):
            gold += self.purchase_resource('LEFT', resources)
            gold += self.purchase_resource('RIGHT', resources)

        if any(resources.values()):
            return None

        return gold

    def purchase_resource(self, side, resources):
        price = 0
        neighbor = self.neighbors[side]
        for resource, quantity in resources.items():
            resource_count = min(neighbor['player'].resources_for_sale[resource], quantity)
            price += resource_count * neighbor['commerce'][resource]
            resources[resource] -= resource_count
        return price

    def apply_effect(self, effect, structure_type):
        if 'gold' in effect:
            self.coins += effect['gold']

        elif 'production' in effect:
            self.productions.append(effect['production'])
            sorted(self.productions, key=lambda p: len(p.keys()))

            if structure_type in [Type.RAW_MATERIAL, Type.MANUFACTURED_GOOD]:
                for resource, quantity in effect['production'].items():
                    self.resources_for_sale[resource] += quantity

        elif 'discount' in effect:
            discount = effect['discount']
            for neighbor in discount['neighbor']:
                for resource in discount['resources']:
                    self.neighbors[neighbor]['commerce'][resource] = discount['price']

        elif 'points' in effect:
            if structure_type == Type.CIVILIAN:
                self.civilian_points += effect['points']
            elif structure_type is None:
                self.wonder_points += effect['points']

        elif 'science' in effect:
            self.scientific_symbols[effect['science']] += 1

        elif 'military' in effect:
            self.shields += effect['military']

        elif 'perBoardElement' in effect and effect['perBoardElement']['gold'] > 0:
            count = self.count_board_elements(effect)
            self.coins += count * effect['perBoardElement']['gold']

        elif 'action' in effect:
            if effect['action'] == 'FREE_BUILD':
                self.free_build_available = True
            elif effect['action'] == 'PLAY_DISCARDED':
                # TODO PLAY_DISCARDED
                pass
            elif effect['action'] == 'PLAY_LAST_CARD':
                # TODO PLAY_LAST_CARD
                pass
            elif effect['action'] == 'COPY_GUILD':
                self.copy_guild = True

    def score(self):
        score = 0
        score += self.victory_points - self.defeat_tokens
        score += floor(self.coins / 3)
        score += self.wonder_points
        score += self.civilian_points

        # TODO choose the "ANY symbol" first
        if self.scientific_symbols:
            score += min(self.scientific_symbols.values()) * 7
        for quantity in self.scientific_symbols.values():
            score += quantity ** 2

        for construction in self.constructions:
            effect = construction['effect']
            if 'perBoardElement' in effect and effect['perBoardElement']['points'] > 0:
                count = self.count_board_elements(effect)
                score += count * effect['perBoardElement']['points']

        # TODO COPY_GUILD

        return score

    def count_board_elements(self, effect):
        count = 0
        if effect['perBoardElement']['type'] == 'CARD':
            for neighbor in effect['perBoardElement']['neighbors']:
                for construction in self.neighbors[neighbor]['player'].constructions:
                    if construction['type'] in effect['perBoardElement']['cardType']:
                        count += 1
        elif effect['perBoardElement']['type'] == 'DEFEAT_TOKEN':
            for neighbor in effect['perBoardElement']['neighbors']:
                count += self.neighbors[neighbor]['player'].defeat_tokens
        elif effect['perBoardElement']['type'] == 'WONDER_STAGES':
            for neighbor in effect['perBoardElement']['neighbors']:
                count += self.neighbors[neighbor]['player'].wonder_stage
        return count

    def resolve_military_conflicts(self, age):
        self.resolve_military_conflicts_with_neighbor(age, self.neighbors['LEFT']['player'])
        self.resolve_military_conflicts_with_neighbor(age, self.neighbors['RIGHT']['player'])

    def resolve_military_conflicts_with_neighbor(self, age, neighbor):
        if neighbor.shields > self.shields:
            self.defeat_tokens += 1
        elif neighbor.shields < self.shields:
            self.victory_points += age * 2 - 1

    def all_productions(self):
        productions = defaultdict(int)
        for production in self.productions:
            for resource, quantity in production.items():
                productions[resource] += quantity

        all_productions = {}
        for resource in ['WOOD', 'STONE', 'CLAY', 'ORE', 'LOOM', 'GLASS', 'PAPYRUS']:
            all_productions[resource] = productions[resource]

        return all_productions


class GameDataJsonDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, data):
        if "__type__" in data:
            return getattr(Type, data["__type__"])
        else:
            return data
