# -*- coding: utf-8 -*-

"""An implementation of the DeepWalk algorithm."""

from typing import Iterable, Optional

from gensim.models import Word2Vec
from igraph import Graph

from .random_walk import RandomWalkParameters, get_random_walks
from .word2vec import Word2VecParameters

__all__ = [
    'get_word2vec',
]


def get_word2vec(graph: Graph,
                 random_walk_parameters: Optional[RandomWalkParameters] = None,
                 word2vec_parameters: Optional[Word2VecParameters] = None) -> Word2Vec:
    """Build a Word2Vec model using random walks on the graph"""
    walks = get_random_walks(
        graph=graph,
        random_walk_parameters=random_walk_parameters,
    )

    # stringify output from igraph for Word2Vec
    walks = (
        map(str, walk)
        for walk in walks
    )

    return _build_word2vec(
        walks=walks,
        word2vec_parameters=word2vec_parameters,
    )


def _build_word2vec(walks: Iterable[Iterable[str]],
                    word2vec_parameters: Optional[Word2VecParameters] = None) -> Word2Vec:
    if word2vec_parameters is None:
        word2vec_parameters = Word2VecParameters()

    return Word2Vec(
        walks,
        size=word2vec_parameters.size,
        window=word2vec_parameters.window,
        min_count=word2vec_parameters.min_count,
        sg=word2vec_parameters.sg,
        hs=word2vec_parameters.hs,
        workers=word2vec_parameters.workers,
    )
