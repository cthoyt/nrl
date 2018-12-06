# -*- coding: utf-8 -*-

"""Algorithms for generating random walks from a given graph."""

import random
from abc import ABC, abstractmethod

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Dict

import numpy as np
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
    restart_probability: Optional[float] = 0.0

    # TODO use this in get_random_walks
    #: random_walk_parameters
    algorithm: Optional[RandomWalkFunction] = None

    #: node2vec parameters
    #: p
    p: Optional[float] = 1.0

    #:q
    q: Optional[float] = 1.0

    # the strategy for sampling the walks
    # TODO: type
    # TODO: implement different strategies
    sampling_strategy: Optional[Dict] = field(default_factory=dict)

    #: Whether the graph is directed or not
    is_directed: Optional[bool] = False

    # Whether the graph is weighted or not
    is_weighted: Optional[bool] = True


class AbstractRandomWalker(ABC):
    def __init__(self, parameters: RandomWalkParameters):
        self.parameters = parameters

    def get_walks(self, graph:Graph):
        for _ in range(self.parameters.number_paths):
            vertices = list(graph.vs)
            random.shuffle(vertices)
            for vertex in graph.vs:
                yield self.get_walk(graph, vertex)

    @abstractmethod
    def get_walk(self, graph: Graph, vertex: Vertex) -> Iterable[Vertex]:
        """Generate one walk."""

def random_walks(graph: Graph,
                 random_walk_parameters: Optional[RandomWalkParameters] = None) -> Iterable[
    Iterable[Vertex]]:
    """Iterate over random walks for all vertices."""
    if random_walk_parameters is None:
        random_walk_parameters = RandomWalkParameters()

    algorithm = random_walk_parameters.algorithm or random_walk_standard

    for _ in range(random_walk_parameters.number_paths):
        vertices = list(graph.vs)
        random.shuffle(vertices)
        for vertex in graph.vs:
            yield algorithm(graph, vertex, random_walk_parameters)


def random_walk_standard(graph: Graph,
                         start: Vertex,
                         params: RandomWalkParameters) -> Iterable[Vertex]:
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
    while path_length < params.max_path_length and graph.neighborhood_size(tail):
        tail = random.choice(tail.neighbors())
        yield tail
        path_length += 1


def random_walk_with_restarts(graph: Graph,
                              start: Vertex,
                              params: RandomWalkParameters) -> Iterable[Vertex]:
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

    while path_length < params.max_path_length and graph.neighborhood_size(tail):
        tail = (
            start
            if params.restart_probability <= random.choice() else
            random.choice(tail.neighbors())
        )
        yield tail
        path_length += 1


def random_walk_normalized():
    """Generate one random walk for one vertex, with the probability for each node inverse to its degree."""


def random_walk_biased(graph: Graph,
                       source: Vertex,
                       params: RandomWalkParameters) -> Iterable[Vertex]:
    """Generates the random walks which will be used as the skip-gram input.

    :return: List of walks. Each walk is a list of nodes.
    """
    # Skip nodes with specific num_walks
    global_walk_length = params.max_path_length
    num_walks = params.number_paths
    sampling_strategy = params.sampling_strategy
    num_walks_key = 'num_walks'
    walk_length_key = 'walk_length'
    probabilities_key = 'probabilities'
    first_travel_key = 'first_travel_key'

    if params.max_path_length < 2:
        raise ValueError("The path length for random walk is less than 2, which doesn't make sense")

    if source in sampling_strategy \
            and num_walks_key in sampling_strategy[source] \
            and sampling_strategy[source][num_walks_key] <= num_walks:
        return

    # Start walk
    yield source
    double_tail = source

    # Calculate walk length
    if source in sampling_strategy:
        walk_length = sampling_strategy[source].get(walk_length_key, global_walk_length)
    else:
        walk_length = global_walk_length

    probabilities = source[first_travel_key]
    tail = np.random.choice(source.neighbors(), p=probabilities)
    if not tail:
        return
    yield tail

    # Perform walk
    path_length = 2
    while path_length < walk_length:
        neighbors = tail.neighbors()

        # Skip dead end nodes
        if not neighbors:
            break

        probabilities = tail[probabilities_key][double_tail['name']]
        double_tail, tail = tail, np.random.choice(neighbors, p=probabilities)

        yield tail
        path_length += 1
