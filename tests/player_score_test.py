import json
import unittest

from game import GameDataJsonDecoder, Player


class PlayerScoreTest(unittest.TestCase):

    def setUp(self):
        with open('../game-data/wonders.json') as wonders_data_file:
            self.wonders = dict((o['name'], o) for o in json.load(wonders_data_file))
        with open('../game-data/age-1-structures.json') as structures_data_file:
            self.structures = dict((o['name'], o) for o in json.load(structures_data_file, cls=GameDataJsonDecoder))
        with open('../game-data/age-2-structures.json') as structures_data_file:
            self.structures_2 = dict((o['name'], o) for o in json.load(structures_data_file, cls=GameDataJsonDecoder))

    def test_initial_score(self):
        left_player = Player({'production': {}})
        right_player = Player({'production': {}})
        player = Player({'production': {}})
        player.with_neighbor(left_player, right_player)
        right_player.with_neighbor(player, left_player)
        left_player.with_neighbor(right_player, player)
        self.assertEqual(player.score(), 1)

    def test_coins_score(self):
        left_player = Player({'production': {}})
        right_player = Player({'production': {}})
        player = Player({'production': {}})
        player.with_neighbor(left_player, right_player)
        right_player.with_neighbor(player, left_player)
        left_player.with_neighbor(right_player, player)
        player.discard_structure()
        self.assertEqual(player.score(), 2)

    def test_wonder_score(self):
        left_player = Player({'production': {}})
        right_player = Player({'production': {}})
        player = Player(self.wonders['Gizah']['sides']['A'])
        player.with_neighbor(left_player, right_player)
        right_player.with_neighbor(player, left_player)
        left_player.with_neighbor(right_player, player)
        player.build_structure(self.structures['Stone Pit'])
        player.build_wonder_stage()
        self.assertEqual(player.score(), 4)


if __name__ == '__main__':
    unittest.main()
