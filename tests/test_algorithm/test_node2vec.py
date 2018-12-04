# -*- coding: utf-8 -*-

"""Tests for Node2Vec."""

import unittest

import networkx as nx
import numpy
from gensim.models import Word2Vec
from node2vec import Node2Vec

from nrl.algorithm.node2vec import run_node2vec
from nrl.algorithm.random_walk import RandomWalkParameters
from nrl.algorithm.word2vec import Word2VecParameters
from nrl.algorithm.node2vec import Node2Vec_temp as NRL_Node2Vec
from tests.constants import get_test_network
from tests.constants import WEIGHTED_NETWORK_PATH


class TestNode2Vec(unittest.TestCase):
    """Test case for DeepWalk."""

    def test_node2vec(self):
        """Test Node2Vec."""
        graph = get_test_network()
        # numpy.random.seed(0)
        # graph.vs["weight"] = list(numpy.random.gamma(2.0, size=len(graph.vs)))  # why node weights?
        # graph.es["weight"] = list(numpy.random.gamma(2.0, size=len(graph.es)))
        #
        # random_walk_parameters = RandomWalkParameters(
        #     number_paths=5,
        #     max_path_length=10,
        # )
        # word2vec_parameters = Word2VecParameters()
        # word2vec = run_node2vec(
        #     graph=graph,
        #     random_walk_parameters=random_walk_parameters,
        #     word2vec_parameters=word2vec_parameters,
        # )
        # self.assertIsInstance(word2vec, Word2Vec)
        #
        # graph = get_test_network(path=WEIGHTED_NETWORK_PATH)

    def test_precompute_probs(self):
        g1 = nx.read_weighted_edgelist(path=WEIGHTED_NETWORK_PATH, nodetype=int)

        n1 = Node2Vec(g1)
        d1 = n1._precompute_probabilities()

        g2 = get_test_network(WEIGHTED_NETWORK_PATH)
        n2 = NRL_Node2Vec(g2)
        d2 = n2._precompute_probabilities()

        for key in d1.keys():
            self.assertListEqual(sorted(d1[key]['neighbors']), sorted(d2[key]['neighbors']))
            self.assertListEqual(list(d1[key]['first_travel_key']), list(d2[key]['first_travel_key']))
            for inner_key in d1[key]['probabilities'].keys():
                self.assertListEqual(
                    list(d1[key]['probabilities'][inner_key]),
                    list(d2[key]['probabilities'][inner_key])
                )



if __name__ == '__main__':
    unittest.main()
