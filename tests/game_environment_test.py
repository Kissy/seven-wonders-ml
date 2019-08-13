import unittest

from environment import GameEnvironment, Action


class GameEnvironmentTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_init(self):
        env = GameEnvironment(3)
        self.assertEqual(env.player_count, 3)
        self.assertEqual(len(env.player_decks), 3)
        self.assertEqual(len(env.player_decks[0]), 7)

        env.step([[Action.BUILD_STRUCTURE, 0], [Action.BUILD_STRUCTURE, 0], [Action.BUILD_STRUCTURE, 0]])


if __name__ == '__main__':
    unittest.main()
