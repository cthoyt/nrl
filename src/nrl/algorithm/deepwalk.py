# -*- coding: utf-8 -*-

"""An implementation of the DeepWalk algorithm."""

from typing import Optional

from gensim.models import Word2Vec
from igraph import Graph

from .word2vec import Word2VecParameters, get_word2vec_from_walks
from ..walker import RandomWalkParameters, StandardRandomWalker

__all__ = [
    'run_deepwalk',
]


def run_deepwalk(graph: Graph,
                 random_walk_parameters: Optional[RandomWalkParameters] = None,
                 word2vec_parameters: Optional[Word2VecParameters] = None) -> Word2Vec:
    """Build a Word2Vec model using random walks on the graph."""
    walker = StandardRandomWalker(random_walk_parameters)
    walks = walker.get_walks(graph)

    # stringify output from igraph for Word2Vec
    walks = (
        map(str, walk)
        for walk in walks
    )

    return get_word2vec_from_walks(
        walks=walks,
        word2vec_parameters=word2vec_parameters,
    )
