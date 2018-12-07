# -*- coding: utf-8 -*-

"""An implementation of the DeepWalk algorithm."""

from typing import Iterable, Optional

from gensim.models import Word2Vec
from igraph import Graph

from .util import BaseModel
from .word2vec import Word2VecParameters, get_word2vec_from_walks
from ..walker import RandomWalkParameters, StandardRandomWalker, Walk

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
    """An implementation of the DeepWalk [1]_ model.

    .. [1] Perozzi, B., Al-Rfou, R., & Skiena, S. (2014). DeepWalk: Online Learning of Social Representations.
           https://doi.org/10.1145/2623330.2623732

    .. seealso:: Other Python implementations of DeepWalk:

        - https://github.com/phanein/deepwalk (reference implementation)
        - https://github.com/thunlp/OpenNE
        - https://github.com/napsternxg/deepwalk_keras_igraph
        - https://github.com/jwplayer/jwalk
    """

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

    def _transform_walks(self, walks: Iterable[Walk]) -> Iterable[Iterable[str]]:
        for walk in walks:
            yield map(str, walk)
