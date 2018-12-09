# -*- coding: utf-8 -*-

"""Utilities for NRL algorithms."""

from abc import ABC, abstractmethod
from typing import Iterable, Optional, Type

from gensim.models import Word2Vec
from igraph import Graph

from .word2vec import Word2VecParameters, get_word2vec_from_walks
from ..typing import Walk
from ..walker import AbstractRandomWalker, RandomWalkParameters

__all__ = [
    'BaseModel',
    'WalkerModel',
]


class BaseModel(ABC):
    """A base model for running Word2Vec-based algorithms."""

    def __init__(self,
                 graph: Graph,
                 random_walk_parameters: Optional[RandomWalkParameters] = None,
                 word2vec_parameters: Optional[Word2VecParameters] = None
                 ) -> None:
        """Store the graph, parameters, then initialize the model."""
        self.graph = graph
        self.random_walk_parameters = random_walk_parameters
        self.word2vec_parameters = word2vec_parameters

        self.initialize()

    @abstractmethod
    def fit(self) -> Word2Vec:
        """Fit the model to the graph and parameters."""

    def initialize(self) -> None:
        """Pre-processing the graph, model, and its parameters."""


class WalkerModel(BaseModel):
    """A base model that uses a random walker to generate walks."""

    random_walker_cls: Type[AbstractRandomWalker]

    def fit(self) -> Word2Vec:
        """Fit the DeepWalk model to the graph and parameters."""
        walker = self.random_walker_cls(self.random_walk_parameters)
        walks = walker.get_walks(self.graph)

        # stringify output from igraph for Word2Vec
        walks = (
            map(str, walk)
            for walk in self.transform_walks(walks)
        )

        return get_word2vec_from_walks(
            walks=walks,
            word2vec_parameters=self.word2vec_parameters,
        )

    def transform_walks(self, walks: Iterable[Walk]) -> Iterable[Walk]:
        """Transform walks (by default, simply returns the walks)."""
        return walks
