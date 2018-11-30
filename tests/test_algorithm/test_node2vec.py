# -*- coding: utf-8 -*-

"""Tests for Node2Vec."""

import unittest

from gensim.models import Word2Vec

from nrl.algorithm.node2vec import run_node2vec
from nrl.algorithm.random_walk import RandomWalkParameters
from nrl.algorithm.word2vec import Word2VecParameters
from tests.constants import get_test_network


class TestNode2Vec(unittest.TestCase):
    """Test case for DeepWalk."""

    def test_node2vec(self):
        """Test Node2Vec."""
        graph = get_test_network()
        random_walk_parameters = RandomWalkParameters(
            number_paths=5,
            max_path_length=10,
        )
        word2vec_parameters = Word2VecParameters()
        word2vec = run_node2vec(
            graph=graph,
            random_walk_parameters=random_walk_parameters,
            word2vec_parameters=word2vec_parameters,
        )
        self.assertIsInstance(word2vec, Word2Vec)
