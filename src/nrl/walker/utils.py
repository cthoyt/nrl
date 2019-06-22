# -*- coding: utf-8 -*-

"""Utilities for random walker algorithms."""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import igraph
import networkx
from dataclasses_json import dataclass_json

from ..typing import Walk, Walks

__all__ = [
    'WalkerParameters',
    'Walker',
]


@dataclass_json
@dataclass
class WalkerParameters:
    """Parameters for random walks."""

    #: The number of paths to get
    number_paths: int = 10

    #: The maximum length the walk can be
    max_path_length: int = 40

    # TODO use this in get_random_walks
    #: Probability of restarting the path. If None, doesn't consider.
    restart_probability: float = 0.0

    """Node2vec parameters"""

    #: p
    p: float = 1.0

    #: q
    q: float = 1.0

    # the strategy for sampling the walks
    # TODO: type
    # TODO: implement different strategies
    sampling_strategy: Optional[Dict] = field(default_factory=dict)

    #: Whether the graph is directed or not
    is_directed: bool = False

    # Whether the graph is weighted or not
    is_weighted: bool = True


class Walker(ABC):
    """An abstract class for random walkers."""

    def __init__(self, parameters: WalkerParameters):
        """Initialize the walker with the given random walk parameters dataclass."""
        self.parameters = parameters

    def get_walks(self, graph: Union[igraph.Graph, networkx.Graph]) -> Walks:
        if isinstance(graph, igraph.Graph):
            return self.get_igraph_walks(graph)
        elif isinstance(graph, networkx.Graph):
            return self.get_networkx_walks(graph)
        else:
            raise TypeError(f'Graph has invalid type: {type(graph)}: {graph}')

    def get_igraph_walks(self, graph: igraph.Graph) -> Walks:
        """Get walks over this graph."""
        for _ in range(self.parameters.number_paths):
            vertices = list(graph.vs)
            random.shuffle(vertices)
            for vertex in vertices:
                yield self.get_igraph_walk(graph, vertex)

    def get_networkx_walks(self, graph: networkx.Graph) -> Walks:
        """Get walks over this graph."""
        for _ in range(self.parameters.number_paths):
            nodes = list(graph)
            random.shuffle(nodes)
            for node in nodes:
                yield self.get_networkx_walk(graph, node)

    @abstractmethod
    def get_igraph_walk(self, graph: igraph.Graph, vertex: igraph.Vertex) -> Walk:
        """Generate one walk."""

    @abstractmethod
    def get_networkx_walk(self, graph: networkx.Graph, vertex: str) -> Walk:
        """Generate one walk."""
