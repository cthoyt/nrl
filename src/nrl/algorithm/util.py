# -*- coding: utf-8 -*-

"""Utilities for NRL algorithms."""

from abc import ABC, abstractmethod
from typing import Optional

from igraph import Graph

from .word2vec import Word2VecParameters
from ..walker import RandomWalkParameters

__all__ = [
    'BaseModel',
]


class BaseModel(ABC):
    """A base model for running Word2Vec-based algorithms."""

    def __init__(self,
                 graph: Graph,
                 random_walk_parameters: Optional[RandomWalkParameters] = None,
                 word2vec_parameters: Optional[Word2VecParameters] = None
                 ) -> None:
        """Initialize the model."""
        self.graph = graph
        self.random_walk_parameters = random_walk_parameters
        self.word2vec_parameters = word2vec_parameters

    @abstractmethod
    def fit(self):
        """Fit the model to the graph and parameters."""
