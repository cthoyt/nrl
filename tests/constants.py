# -*- coding: utf-8 -*-

"""Testing constants for `nrl`."""

import os

import igraph

from nrl.io import read_ncol_graph

__all__ = [
    'HERE',
    'TEST_NETWORK_PATH',
    'get_test_network',
]

HERE = os.path.abspath(os.path.dirname(__file__))
TEST_NETWORK_PATH = os.path.join(HERE, 'hippie_current.edgelist')


def get_test_network() -> igraph.Graph:
    """Get the test network."""
    return read_ncol_graph(TEST_NETWORK_PATH)
