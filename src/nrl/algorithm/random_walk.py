# -*- coding: utf-8 -*-

"""Algorithms for generating random walks from a given graph."""

import random
from dataclasses import dataclass

from igraph import Graph, Vertex


@dataclass
class RandomWalkParameters:
    """"""


def get_random_walks(graph: Graph, number_paths: int, max_path_length: int):
    """"""
    return list(_get_random_walks_iter(
        graph=graph,
        number_paths=number_paths,
        max_path_length=max_path_length,
    ))


def get_random_walk(graph: Graph, start: Vertex, length: int):
    """Generate one random walk for one node.

    :param graph: The graph to investigate
    :param start: The vertex at which the random walk starts
    :param length: The length of the path to get
    """
    path = [start]

    while len(path) < length:
        tail = path[-1]

        if graph.neighborhood_size(tail) == 0:  # return the current path if there are no neighbors
            return path

        path.append(random.choice(graph.neighborhood(tail)))

    return path


def _get_random_walks_iter(graph: Graph, number_paths: int, max_path_length: int):
    """Get random walks for all nodes."""
    for _ in range(number_paths):
        nodes = list(graph.vs)
        random.shuffle(nodes)
        for node in graph.vs:
            yield get_random_walk(graph, node, max_path_length)


def get_random_walk_with_restart(graph: Graph, start: Vertex, max_path_length: int, alpha: float = 0.0):
    """Generate one random walk for one node, with the probability alpha of restarting.

    :param graph: The graph to investigate
    :param start: The vertex at which the random walk starts
    :param max_path_length: The length of the path to get
    :param alpha: Probability of restart.
    """
    path = [start]

    while len(path) < max_path_length:
        tail = path[-1]

        if graph.neighborhood_size(tail) == 0:  # return the current path if there are no neighbors
            return path

        path.append(
            start
            if alpha <= random.choice() else
            random.choice(graph.neighborhood(tail))
        )

    return path
