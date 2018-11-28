# -*- coding: utf-8 -*-

"""Algorithms for generating random walks from a given graph."""

import random
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from igraph import Graph, Vertex

RandomWalkFunction = Callable[[Graph, Vertex, int], Iterable[Vertex]]


@dataclass
class RandomWalkParameters:
    """Parameters for random walks."""

    #: The number of paths to get
    number_paths: int = 10

    #: The maximum length the walk can be
    max_path_length: int = 40

    # TODO use this in get_random_walks
    #: Probability of restarting the path. If None, doesn't consider.
    restart_probability: Optional[float] = None

    # TODO use this in get_random_walks
    #: random_walk_parameters
    algorithm: Optional[RandomWalkFunction] = None


def random_walks(graph: Graph,
                 random_walk_parameters: Optional[RandomWalkParameters] = None) -> Iterable[Iterable[Vertex]]:
    """Iterate over random walks for all vertices."""
    if random_walk_parameters is None:
        random_walk_parameters = RandomWalkParameters()

    algorithm = random_walk_parameters.algorithm or random_walk_standard

    for _ in range(random_walk_parameters.number_paths):
        vertices = list(graph.vs)
        random.shuffle(vertices)
        for vertex in graph.vs:
            yield algorithm(graph, vertex, random_walk_parameters.max_path_length)


def random_walk_standard(graph: Graph,
                         start: Vertex,
                         max_path_length: int) -> Iterable[Vertex]:
    """Generate one random walk for one vertex.

    :param graph: The graph to investigate
    :param start: The vertex at which the random walk starts
    :param max_path_length: The length of the path to get. If running into a vertex without enough neighbors, will
     return a shorter path.
    """
    tail = start
    yield tail
    path_length = 1

    # return if the the current path is too long or there if there are no neighbors at the end
    while path_length < max_path_length and graph.neighborhood_size(tail) != 0:
        tail = random.choice(graph.neighborhood(tail))
        yield tail
        path_length += 1


def random_walk_with_restarts(graph: Graph,
                              start: Vertex,
                              max_path_length: int,
                              alpha: float = 0.0) -> Iterable[Vertex]:
    """Generate one random walk for one vertex, with the probability, alpha, of restarting.

    :param graph: The graph to investigate
    :param start: The vertex at which the random walk starts
    :param max_path_length: The length of the path to get. If running into a vertex without enough neighbors, will
     return a shorter path.
    :param alpha: Probability of restart.
    """
    tail = start
    yield tail
    path_length = 1

    while path_length < max_path_length and graph.neighborhood_size(tail) != 0:
        tail = (
            start
            if alpha <= random.choice() else
            random.choice(graph.neighborhood(tail))
        )
        yield tail
        path_length += 1


def random_walk_normalized():
    """Generate one random walk for one vertex, with the probability for each node inverse to its degree."""
