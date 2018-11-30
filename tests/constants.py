# -*- coding: utf-8 -*-

"""Testing constants for `nrl`."""

import os

import igraph

from nrl.io import read_ncol_graph

__all__ = [
    'HERE',
    'TEST_HIPPIE_NETWORK_PATH',
    'KARATE_CLUB_PATH',
    'get_test_network',
]

HERE = os.path.abspath(os.path.dirname(__file__))

RESOURCES_DIRECTORY = os.path.join(HERE, 'resources')
KARATE_CLUB_PATH = os.path.join(RESOURCES_DIRECTORY, 'karate.edgelist')
TEST_HIPPIE_NETWORK_PATH = os.path.join(RESOURCES_DIRECTORY, 'hippie_current.edgelist')


def get_test_network() -> igraph.Graph:
    """Get the test network."""
    return read_ncol_graph(KARATE_CLUB_PATH)
