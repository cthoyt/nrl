# -*- coding: utf-8 -*-

"""Implementations of random walk algorithms."""

import random

import numpy as np
from igraph import Graph, Vertex

from .utils import AbstractRandomWalker
from ..typing import Walk

__all__ = [
    'StandardRandomWalker',
    'RestartingRandomWalker',
    'BiasedRandomWalker',
]


class StandardRandomWalker(AbstractRandomWalker):
    """Make standard random walks, choosing the neighbors at a given position uniformly."""

    def get_walk(self, graph: Graph, vertex: Vertex) -> Walk:
        """Get a random walk by choosing from the neighbors at a given position uniformly."""
        tail = vertex
        yield tail
        path_length = 1
        # return if the the current path is too long or there if there are no neighbors at the end
        while path_length < self.parameters.max_path_length and graph.neighborhood_size(tail):
            tail = random.choice(tail.neighbors())
            yield tail
            path_length += 1


class RestartingRandomWalker(AbstractRandomWalker):
    """A random walker that restarts from the original vertex with a given probability."""

    @property
    def restart_probability(self) -> float:
        """Get the probability with which this walker will restart from the original vertex."""
        return self.parameters.restart_probability

    def get_walk(self, graph: Graph, vertex: Vertex) -> Walk:
        """Generate one random walk for one vertex, with the probability, alpha, of restarting."""
        tail = vertex
        yield tail
        path_length = 1

        while path_length < self.parameters.max_path_length and graph.neighborhood_size(tail):
            tail = (
                vertex
                if self.restart_probability <= random.choice() else
                random.choice(tail.neighbors())
            )
            yield tail
            path_length += 1


class BiasedRandomWalker(AbstractRandomWalker):
    """A random walker that generates second-order random walks biased by edge weights."""

    def get_walk(self, graph: Graph, vertex: Vertex) -> Walk:
        """Generate second-order random walks biased by edge weights."""
        num_walks_key = 'num_walks'
        walk_length_key = 'walk_length'
        probabilities_key = 'probabilities'
        first_travel_key = 'first_travel_key'

        global_walk_length = self.parameters.max_path_length
        num_walks = self.parameters.number_paths
        sampling_strategy = self.parameters.sampling_strategy

        if self.parameters.max_path_length < 2:
            raise ValueError("The path length for random walk is less than 2, which doesn't make sense")

        if vertex in sampling_strategy \
                and num_walks_key in sampling_strategy[vertex] \
                and sampling_strategy[vertex][num_walks_key] <= num_walks:
            return

        # Start walk
        yield vertex
        double_tail = vertex

        # Calculate walk length
        if vertex in sampling_strategy:
            walk_length = sampling_strategy[vertex].get(walk_length_key, global_walk_length)
        else:
            walk_length = global_walk_length

        probabilities = vertex[first_travel_key]
        tail = np.random.choice(vertex.neighbors(), p=probabilities)
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
