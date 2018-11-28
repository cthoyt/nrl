# -*- coding: utf-8 -*-

"""Test for I/O utilities."""

import unittest

from tests.constants import get_test_network


class TestIO(unittest.TestCase):
    """Test case for I/O utilities."""

    def test_read_graph(self):
        """Test reading a graph."""
        graph = get_test_network()
        self.assertEqual(16829, len(graph.vs))
        self.assertEqual(287357, len(graph.es))
