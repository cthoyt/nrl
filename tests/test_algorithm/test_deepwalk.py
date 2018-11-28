# -*- coding: utf-8 -*-

"""Tests for DeepWalk."""

import unittest

from gensim.models import Word2Vec

from nrl.algorithm.deepwalk import get_word2vec
from tests.constants import get_test_network


class TestDeepWalk(unittest.TestCase):
    """Test case for DeepWalk."""

    def test_deepwalk(self):
        """Test DeepWalk."""
        graph = get_test_network()
        word2vec = get_word2vec(graph)
        self.assertIsInstance(word2vec, Word2Vec)
