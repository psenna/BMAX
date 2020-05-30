import unittest

class Databases_test_suit(unittest.TestCase):

    def test_get_minist(self):
        from src.databases import Databases

        images, labels = Databases.get_minist('../')

