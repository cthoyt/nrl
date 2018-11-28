# -*- coding: utf-8 -*-

"""Algorithms for generating random walks from a given graph."""

import random
from dataclasses import dataclass
from typing import Iterable, List, Optional

from igraph import Graph, Vertex


@dataclass
class RandomWalkParameters:
    """"""

    #: The number of paths to get
    number_paths: int = 10

    #:
    max_path_length: int = 40

    # TODO use this in get_random_walks
    #: Probability of restarting the path. If None, doesn't consider.
    restart_probability: Optional[float] = None

    # TODO use this in get_random_walks
    #: random_walk_parameters
    algorithm: str = 'standard'


def get_random_walks(graph: Graph,
                     random_walk_parameters: Optional[RandomWalkParameters] = None) -> Iterable[List[Vertex]]:
    """"""
    if random_walk_parameters is None:
        random_walk_parameters = RandomWalkParameters()

    return _get_random_walks_iter(
        graph=graph,
        random_walk_parameters=random_walk_parameters,
    )


def _get_random_walks_iter(graph: Graph,
                           random_walk_parameters: Optional[RandomWalkParameters] = None) -> Iterable[List[Vertex]]:
    """Get random walks for all nodes."""
    if random_walk_parameters is None:
        random_walk_parameters = RandomWalkParameters()

    for _ in range(random_walk_parameters.number_paths):
        nodes = list(graph.vs)
        random.shuffle(nodes)
        for node in graph.vs:
            yield get_random_walk(graph, node, random_walk_parameters.max_path_length)


def get_random_walk(graph: Graph, start: Vertex, length: int) -> List[Vertex]:
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
