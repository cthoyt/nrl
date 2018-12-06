# -*- coding: utf-8 -*-

"""An implementation of the GAT2VEC extension of the DeepWalk algorithm."""

from typing import Optional

from gensim.models import Word2Vec
from igraph import Graph, VertexSeq

from .word2vec import Word2VecParameters, get_word2vec_from_walks
from ..walker import RandomWalkParameters, StandardRandomWalker


def run_gat2vec_unsupervised(graph: Graph,
                             structural_vertices: VertexSeq,
                             random_walk_parameters: Optional[RandomWalkParameters] = None,
                             word2vec_parameters: Optional[Word2VecParameters] = None) -> Word2Vec:
    """Run the unsupervised GAT2VEC algorithm.

    :param graph:
    :param structural_vertices:
    :param random_walk_parameters:
    :param word2vec_parameters:
    """
    # double the maximum path length, because GAT2VEC used this as the heuristic
    random_walk_parameters.max_path_length *= 2

    walker = StandardRandomWalker(random_walk_parameters)
    walks = walker.get_walks(graph)

    # stringify output from igraph for Word2Vec
    # filter out vertices that aren't structural vertices
    walks = (
        (
            str(vertex)
            for vertex in walk
            if vertex in structural_vertices
        )
        for walk in walks
    )

    return get_word2vec_from_walks(
        walks=walks,
        word2vec_parameters=word2vec_parameters,
    )
