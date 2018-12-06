# -*- coding: utf-8 -*-

"""An implementation of the DeepWalk algorithm."""

from typing import Iterable, Optional

from gensim.models import Word2Vec
from igraph import Graph, Vertex

from .util import BaseModel
from .word2vec import Word2VecParameters, get_word2vec_from_walks
from ..walker import RandomWalkParameters, StandardRandomWalker

__all__ = [
    'run_deepwalk',
    'DeepWalkModel',
]


def run_deepwalk(graph: Graph,
                 random_walk_parameters: Optional[RandomWalkParameters] = None,
                 word2vec_parameters: Optional[Word2VecParameters] = None
                 ) -> Word2Vec:
    """Build a Word2Vec model using random walks on the graph."""
    model = DeepWalkModel(
        graph=graph,
        random_walk_parameters=random_walk_parameters,
        word2vec_parameters=word2vec_parameters,
    )
    return model.fit()


class DeepWalkModel(BaseModel):
    """An implementation of the DeepWalk model."""

    def fit(self):
        """Fit the DeepWalk model to the graph and parameters."""
        walker = StandardRandomWalker(self.random_walk_parameters)
        walks = walker.get_walks(self.graph)

        # stringify output from igraph for Word2Vec
        walks = self._transform_walks(walks)

        return get_word2vec_from_walks(
            walks=walks,
            word2vec_parameters=self.word2vec_parameters,
        )

    def _transform_walks(self, walks: Iterable[Iterable[Vertex]]) -> Iterable[Iterable[str]]:
        for walk in walks:
            yield map(str, walk)
