# -*- coding: utf-8 -*-

"""Test for I/O utilities."""

import unittest

from tests.constants import get_test_network


class TestIO(unittest.TestCase):
    """Test case for I/O utilities."""

    def test_read_karate_club_graph(self):
        """Test reading the karate club graph."""
        graph = get_test_network()
        self.assertEqual(34, len(graph.vs))
        self.assertEqual(78, len(graph.es))
