# -*- coding: utf-8 -*-

"""Algorithms for generating random walks for Node2vec."""

import random
from typing import List, Optional

import numpy as np
from gensim.models import Word2Vec
from igraph import Graph, Vertex

from .random_walk import RandomWalkParameters
from .word2vec import Word2VecParameters, get_word2vec_from_walks

__all__ = [
    'run_node2vec',
]

WEIGHT = 'weight'


def run_node2vec(graph: Graph,
                 random_walk_parameters: Optional[RandomWalkParameters] = None,
                 word2vec_parameters: Optional[Word2VecParameters] = None) -> Word2Vec:
    """Run node2vec."""
    walker = Node2VecWalker(graph=graph, is_directed=False, p=1.0, q=1.0)
    walks = walker.simulate_walks(
        walk_length=random_walk_parameters.max_path_length,
        num_walks=random_walk_parameters.number_paths,
    )

    return get_word2vec_from_walks(
        walks,
        word2vec_parameters=word2vec_parameters
    )


class Node2VecWalker:
    """Create walks using the Node2Vec algorithm for use with Word2Vec."""

    def __init__(self, graph: Graph, is_directed: bool, p: float, q: float):
        self.graph = graph
        self.is_directed = is_directed
        self.p = p
        self.q = q

        self.alias_nodes = self._get_alias_nodes(graph)

        if is_directed:
            self.alias_edges = self._get_directed_alias_edges(graph, p, q)
        else:
            self.alias_edges = self._get_undirected_alias_edges(graph, p, q)

    def node2vec_walk(self, walk_length: int, start: Vertex):
        """Simulate a random walk starting from start node."""
        walk = [start]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_neighbors = sorted(self.graph.neighbors(cur))
            if 0 == len(cur_neighbors):
                break
            if len(walk) == 1:
                walk.append(cur_neighbors[alias_draw(self.alias_nodes[cur][0], self.alias_nodes[cur][1])])
            else:
                prev = walk[-2]
                next = cur_neighbors[alias_draw(self.alias_edges[prev, cur][0], self.alias_edges[prev, cur][1])]
                walk.append(next)

        return walk

    def simulate_walks(self, num_walks: int, walk_length: int):
        """Repeatedly simulate random walks from each node."""
        walks = []
        nodes = list(self.graph.vs)
        for _ in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walk = self.node2vec_walk(
                    walk_length=walk_length,
                    start=node,
                )
                walks.append(walk)

        return walks

    @staticmethod
    def _get_alias_nodes(graph: Graph):
        alias_nodes = {}

        for vertex in graph.vs:
            neighbor_weights = [
                graph[vertex][neighbor][WEIGHT]
                for neighbor in sorted(graph.neighborhood(vertex))
            ]
            normalized_neighbor_weights = normalize(neighbor_weights)
            alias_nodes[vertex] = alias_setup(normalized_neighbor_weights)

        return alias_nodes

    @staticmethod
    def _get_directed_alias_edges(graph: Graph, p: float, q: float):
        return {
            (source, target): get_alias_edge(graph, source, target, p, q)
            for source, target in graph.es
        }

    @staticmethod
    def _get_undirected_alias_edges(graph, p: float, q: float):
        alias_edges = {}

        for source, target in graph.es:
            alias_edges[source, target] = get_alias_edge(graph, source, target, p, q)
            alias_edges[target, source] = get_alias_edge(graph, target, source, p, q)

        return alias_edges


def get_alias_edge(graph: Graph, source: Vertex, target: Vertex, p: float, q: float):
    """Get the alias edge setup lists for a given edge."""
    unnormalized_probs = []

    for target_neighbor in sorted(graph.neighbors(target)):
        if target_neighbor == source:
            unnormalized_probs.append(graph[target][target_neighbor][WEIGHT] / p)
        elif graph.has_edge(target_neighbor, source):
            unnormalized_probs.append(graph[target][target_neighbor][WEIGHT])
        else:
            unnormalized_probs.append(graph[target][target_neighbor][WEIGHT] / q)

    normalized_probs = normalize(unnormalized_probs)

    return alias_setup(normalized_probs)


def normalize(numbers: List[float]) -> List[float]:
    total = sum(numbers)
    return [number / total for number in numbers]


def alias_setup(probs: List[float]):
    """Compute utility lists for non-uniform sampling from discrete distributions.

    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    """Draw sample from a non-uniform discrete distribution using alias sampling."""
    kk = np.random.randint(len(J))

    if np.random.rand() < q[kk]:
        return kk

    return J[kk]
