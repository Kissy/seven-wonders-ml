import json
import unittest

from game import GameDataJsonDecoder, Player


class PlayerBuildCostTest(unittest.TestCase):

    def setUp(self):
        with open('../game-data/wonders.json') as wonders_data_file:
            self.wonders = dict((o['name'], o) for o in json.load(wonders_data_file))
        with open('../game-data/age-1-structures.json') as structures_data_file:
            self.structures = dict((o['name'], o) for o in json.load(structures_data_file, cls=GameDataJsonDecoder))
        with open('../game-data/age-2-structures.json') as structures_data_file:
            self.structures_2 = dict((o['name'], o) for o in json.load(structures_data_file, cls=GameDataJsonDecoder))

    def test_that_construction_is_not_possible_with_not_enough_productions(self):
        right_player = Player({'production': {}})
        left_player = Player({'production': {}})
        player = Player({'production': {}})
        player.with_neighbor(right_player, left_player)
        self.assertEqual(player.build_cost(self.structures['Baths']['cost']), None)

    def test_that_construction_is_possible_with_enough_productions(self):
        right_player = Player({'production': {}})
        left_player = Player({'production': {}})
        player = Player({'production': {}})
        player.with_neighbor(right_player, left_player)
        player.build_structure(self.structures['Stone Pit'])
        self.assertEqual(player.build_cost(self.structures['Baths']['cost']), 0)

    def test_that_construction_is_possible_with_right_commerce(self):
        left_player = Player({'production': {}})
        right_player = Player({'production': {}})
        player = Player({'production': {}})
        player.with_neighbor(left_player, right_player)
        right_player.with_neighbor(player, left_player)
        left_player.with_neighbor(right_player, player)
        right_player.build_structure(self.structures['Stone Pit'])
        self.assertEqual(player.build_cost(self.structures['Baths']['cost']), 2)

    def test_that_construction_is_possible_with_left_commerce(self):
        left_player = Player({'production': {}})
        right_player = Player({'production': {}})
        player = Player({'production': {}})
        player.with_neighbor(left_player, right_player)
        right_player.with_neighbor(player, left_player)
        left_player.with_neighbor(right_player, player)
        left_player.build_structure(self.structures['Stone Pit'])
        self.assertEqual(player.build_cost(self.structures['Baths']['cost']), 2)

    def test_that_construction_is_possible_with_reduced_commerce(self):
        left_player = Player({'production': {}})
        right_player = Player({'production': {}})
        player = Player({'production': {}})
        player.with_neighbor(left_player, right_player)
        right_player.with_neighbor(player, left_player)
        left_player.with_neighbor(right_player, player)
        right_player.build_structure(self.structures['Stone Pit'])
        player.build_structure(self.structures['East Trading Post'])
        self.assertEqual(player.build_cost(self.structures['Baths']['cost']), 1)

    def test_that_construction_is_possible_with_chaining(self):
        left_player = Player({'production': {}})
        right_player = Player({'production': {}})
        player = Player({'production': {}})
        player.with_neighbor(left_player, right_player)
        right_player.with_neighbor(player, left_player)
        left_player.with_neighbor(right_player, player)
        player.build_structure(self.structures['Theater'])
        self.assertEqual(player.build_cost(self.structures_2['Statue']['cost']), 0)


if __name__ == '__main__':
    unittest.main()
