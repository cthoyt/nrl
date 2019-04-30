# -*- coding: utf-8 -*-

"""An implementation of the DeepWalk algorithm."""

from typing import Optional

from gensim.models import Word2Vec
from igraph import Graph

from .util import WalkerModel
from .word2vec import Word2VecParameters
from ..walker import StandardRandomWalker, WalkerParameters

__all__ = [
    'run_deepwalk',
    'DeepWalkModel',
]


def run_deepwalk(
        graph: Graph,
        walker_parameters: Optional[WalkerParameters] = None,
        word2vec_parameters: Optional[Word2VecParameters] = None
) -> Word2Vec:
    """Run the DeepWalk algorithm to generate a Word2Vec model."""
    model = DeepWalkModel(
        walker_parameters=walker_parameters,
        word2vec_parameters=word2vec_parameters,
    )
    return model.fit(graph)


class DeepWalkModel(WalkerModel):
    """An implementation of the DeepWalk model [perozzi2014]_.

    .. [perozzi2014] Perozzi, B., Al-Rfou, R., & Skiena, S. (2014). DeepWalk: Online Learning of Social Representations.
                     https://doi.org/10.1145/2623330.2623732

    .. seealso:: Other Python implementations of DeepWalk:

        - https://github.com/phanein/deepwalk (reference implementation)
        - https://github.com/thunlp/OpenNE
        - https://github.com/napsternxg/deepwalk_keras_igraph
        - https://github.com/jwplayer/jwalk
    """

    walker_cls = StandardRandomWalker
