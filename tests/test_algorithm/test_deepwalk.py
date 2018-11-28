# -*- coding: utf-8 -*-

"""Tests for DeepWalk."""

import unittest

from gensim.models import Word2Vec

from nrl.algorithm.deepwalk import get_word2vec
from nrl.algorithm.random_walk import RandomWalkParameters
from nrl.algorithm.word2vec import Word2VecParameters
from tests.constants import get_test_network


class TestDeepWalk(unittest.TestCase):
    """Test case for DeepWalk."""

    def test_deepwalk(self):
        """Test DeepWalk."""
        graph = get_test_network()
        random_walk_parameters = RandomWalkParameters(
            number_paths=5,
            max_path_length=10,
        )
        word2vec_parameters = Word2VecParameters()
        word2vec = get_word2vec(
            graph=graph,
            random_walk_parameters=random_walk_parameters,
            word2vec_parameters=word2vec_parameters,
        )
        self.assertIsInstance(word2vec, Word2Vec)
