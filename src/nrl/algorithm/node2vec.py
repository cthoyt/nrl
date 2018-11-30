# -*- coding: utf-8 -*-

"""Algorithms for generating random walks for Node2vec."""

import random
from typing import List, Mapping, Optional, Tuple

import numpy as np
from gensim.models import Word2Vec
from igraph import Graph, Vertex

from .random_walk import RandomWalkParameters
from .word2vec import Word2VecParameters, get_word2vec_from_walks

__all__ = [
    'run_node2vec',
]

WEIGHT = 'weight'

AliasData = Tuple[np.ndarray, np.ndarray]


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

    def __init__(self,
                 graph: Graph,
                 is_directed: bool = False,
                 p: float = 1.0,
                 q: float = 1.0) -> None:
        """Initialize the walker.

        :param graph:
        :param is_directed:
        :param p: The return hyper-parameter
        :param q: The input parameter
        """
        self.graph = graph
        self.is_directed = is_directed
        self.p = p
        self.q = q

        self.alias_nodes: Mapping[int, AliasData] = self._get_alias_nodes(graph)
        self.alias_edges: Mapping[Tuple[int, int], AliasData] = (
            self._get_directed_alias_edges(graph, p, q)
            if is_directed else
            self._get_undirected_alias_edges(graph, p, q)
        )

    def node2vec_walk(self, walk_length: int, start: int) -> List[int]:
        """Simulate a random walk starting from start node."""
        walk: List[int] = [start]

        if 0 == self.graph.neighborhood_size(start):
            return walk  # this start node has no neighbors

        j, q = self.alias_nodes[start]
        print()
        print('J', j)
        print('q', q)

        start_neighbor_idx = alias_draw(j, q)
        print('start index', start_neighbor_idx)

        start_neighbors = sorted(self.graph.neighbors(start))
        print('start neighbors', start_neighbors)
        walk.append(start_neighbors[start_neighbor_idx])

        while len(walk) < walk_length:
            current_vertex = walk[-1]

            if 0 == self.graph.neighborhood_size(current_vertex):
                break

            prev = walk[-2]
            j, q = self.alias_edges[prev, current_vertex]
            idx = alias_draw(j, q)

            cur_neighbors = sorted(self.graph.neighbors(current_vertex))
            walk.append(cur_neighbors[idx])

        return walk

    def simulate_walks(self, num_walks: int, walk_length: int):
        """Repeatedly simulate random walks from each node."""
        walks = []
        vertices = list(self.graph.vs)
        for _ in range(num_walks):
            random.shuffle(vertices)
            for vertex in vertices:
                walk = self.node2vec_walk(
                    walk_length=walk_length,
                    start=vertex.index,
                )
                walks.append(walk)

        return walks

    @staticmethod
    def _get_alias_nodes(graph: Graph) -> Mapping[int, AliasData]:
        return {
            vertex.index: alias_setup([
                graph.vs[neighbor][WEIGHT]
                for neighbor in sorted(graph.neighborhood(vertex))
            ])
            for vertex in graph.vs
        }

    @staticmethod
    def _get_directed_alias_edges(graph: Graph, p: float, q: float) -> Mapping[Tuple[int, int], AliasData]:
        return {
            (source, target): get_alias_edge(graph, source, target, p, q)
            for source, target in graph.es
        }

    @staticmethod
    def _get_undirected_alias_edges(graph: Graph, p: float, q: float) -> Mapping[Tuple[int, int], AliasData]:
        alias_edges = {}

        for edge in graph.es:
            source, target = edge.source, edge.target
            alias_edges[source, target] = get_alias_edge(graph, source, target, p, q)
            alias_edges[target, source] = get_alias_edge(graph, target, source, p, q)

        return alias_edges


def get_alias_edge(graph: Graph, source: Vertex, target: Vertex, p: float, q: float):
    """Get the alias edge setup lists for a given edge."""
    probs = []

    for target_neighbor in sorted(graph.neighbors(target)):
        weight = graph.es[graph.get_eid(target, target_neighbor)][WEIGHT]

        if target_neighbor == source:
            prob = weight / p
        elif graph.are_connected(target_neighbor, source):
            prob = weight
        else:
            prob = weight / q

        probs.append(prob)

    return alias_setup(probs)


def normalize(numbers: List[float]) -> List[float]:
    total = sum(numbers)
    return [number / total for number in numbers]


def alias_setup(probs: List[float]) -> AliasData:
    """Compute utility lists for non-uniform sampling from discrete distributions.

    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    """
    probs = normalize(probs)
    K = len(probs)
    q = np.zeros(K)
    j = np.zeros(K, dtype=np.int)

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

        j[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return j, q


def alias_draw(j, q):
    """Draw sample from a non-uniform discrete distribution using alias sampling."""
    kk = np.random.randint(len(j))

    if np.random.rand() < q[kk]:
        return kk

    return j[kk]
