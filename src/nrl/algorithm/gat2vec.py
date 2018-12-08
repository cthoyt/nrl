# -*- coding: utf-8 -*-

"""An implementation of the GAT2VEC extension of the DeepWalk algorithm."""

from typing import Iterable, Optional

from gensim.models import Word2Vec
from igraph import Graph, VertexSeq

from .util import WalkerModel
from .word2vec import Word2VecParameters
from ..typing import Walk
from ..walker import RandomWalkParameters, StandardRandomWalker

__all__ = [
    'run_gat2vec_unsupervised',
    'Gat2VecUnsupervisedModel',
]


def run_gat2vec_unsupervised(graph: Graph,
                             structural_vertices: VertexSeq,
                             random_walk_parameters: Optional[RandomWalkParameters] = None,
                             word2vec_parameters: Optional[Word2VecParameters] = None
                             ) -> Word2Vec:
    """Run the unsupervised GAT2VEC algorithm to generate a Word2Vec model."""
    model = Gat2VecUnsupervisedModel(
        graph=graph,
        structural_vertices=structural_vertices,
        random_walk_parameters=random_walk_parameters,
        word2vec_parameters=word2vec_parameters,
    )
    return model.fit()


class Gat2VecUnsupervisedModel(WalkerModel):
    """An implementation of the GAT2VEC unsupervised model [sheikh2018]_ .

    .. [sheikh2018] Sheikh, N., Kefato, Z., & Montresor, A. (2018). Gat2Vec: Representation Learning for Attributed
                    Graphs. Computing, 1–23. https://doi.org/10.1007/s00607-018-0622-9

    .. seealso:: Other Python implementations of GAT2VEC:

        - https://github.com/snash4/GAT2VEC (reference implementation)
    """

    random_walker_cls = StandardRandomWalker

    def __init__(self,
                 graph: Graph,
                 structural_vertices: VertexSeq,
                 random_walk_parameters: Optional[RandomWalkParameters] = None,
                 word2vec_parameters: Optional[Word2VecParameters] = None
                 ) -> None:
        """Initialize the GAT2VEC unsupervised model."""
        super().__init__(
            graph=graph,
            random_walk_parameters=random_walk_parameters,
            word2vec_parameters=word2vec_parameters,
        )

        # Double the maximum path length since the paths will be filtered
        self.random_walk_parameters.max_path_length *= 2

        # Store structural vertices - other ones will be filtered out
        self.structural_vertices = structural_vertices

    def _transform_walks(self, walks: Iterable[Walk]) -> Iterable[Walk]:
        for walk in walks:
            yield (
                vertex
                for vertex in walk
                if vertex in self.structural_vertices
            )
