# -*- coding: utf-8 -*-

"""An implementation of the GAT2VEC extension of the DeepWalk algorithm."""

from typing import Optional

from gensim.models import Word2Vec
from igraph import Graph, VertexSeq

from .deepwalk import _build_word2vec
from .word2vec import Word2VecParameters
from .random_walk import RandomWalkParameters, get_random_walks


def do_gat2vec_unsupervized(graph: Graph,
                            structural_vertices: VertexSeq,
                            random_walk_parameters: Optional[RandomWalkParameters] = None,
                            word2vec_parameters: Optional[Word2VecParameters] = None) -> Word2Vec:
    """

    :param graph:
    :param structural_vertices:
    :param random_walk_parameters:
    :param word2vec_parameters:
    :return:
    """
    # double the maximum path length, because GAT2VEC used this as the heuristic
    random_walk_parameters.max_path_length *= 2

    walks = get_random_walks(
        graph=graph,
        random_walk_parameters=random_walk_parameters,
    )

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

    return _build_word2vec(
        walks=walks,
        word2vec_parameters=word2vec_parameters,
    )
