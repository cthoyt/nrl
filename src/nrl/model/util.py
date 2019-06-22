# -*- coding: utf-8 -*-

"""Utilities for NRL algorithms."""

import json
from abc import ABC, abstractmethod
from typing import Optional, Type

from gensim.models import Word2Vec

from .word2vec import Word2VecParameters, get_word2vec_from_walks
from ..constants import get_version
from ..typing import Graph, Walks
from ..walker import Walker, WalkerParameters

__all__ = [
    'BaseModel',
    'WalkerModel',
]


class BaseModel(ABC):
    """A base model for running Word2Vec-based algorithms."""

    def __init__(
            self,
            walker_parameters: Optional[WalkerParameters] = None,
            word2vec_parameters: Optional[Word2VecParameters] = None
    ) -> None:
        """Store the graph, parameters, then initialize the model."""
        self.random_walk_parameters = walker_parameters or WalkerParameters()
        self.word2vec_parameters = word2vec_parameters or Word2VecParameters()

        # Model is saved after being fit
        self.model: Optional[Word2Vec] = None

        self.initialize()

    @abstractmethod
    def fit(self, graph: Graph) -> Word2Vec:
        """Fit the model to the graph and parameters."""

    def initialize(self) -> None:
        """Pre-processing the graph, model, and its parameters."""

    def get_metadata(self):
        """Get metadata for this model."""
        return {
            'package_name': 'nrl',
            'package_version': get_version(),
            # TODO get losses or training info from word2vec
            'parameters': {
                'random_walk': self.random_walk_parameters.to_json(),
                'word2vec': self.word2vec_parameters.to_json(),
            },
        }

    def dump_metadata(self, path: str, *, indent: int = 2, **kwargs) -> None:
        """Dump metadata for this model to a file."""
        with open(path, 'w') as file:
            json.dump(self.get_metadata(), file, indent=indent, **kwargs)


class WalkerModel(BaseModel):
    """A base model that uses a random walker to generate walks."""

    walker_cls: Type[Walker]

    def fit(self, graph: Graph) -> Word2Vec:
        """Fit the DeepWalk model to the graph and parameters."""
        walker = self.walker_cls(self.random_walk_parameters)
        walks = walker.get_walks(graph)
        walks = self.transform_walks(walks)

        self.model = get_word2vec_from_walks(
            walks=walks,
            word2vec_parameters=self.word2vec_parameters,
        )
        return self.model

    def transform_walks(self, walks: Walks) -> Walks:
        """Transform walks (by default, simply returns the walks)."""
        return walks
