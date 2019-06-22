# -*- coding: utf-8 -*-

"""An implementation of the GAT2VEC extension of the DeepWalk algorithm."""

from typing import Iterable, Optional, Union

from gensim.models import Word2Vec
from igraph import Graph, VertexSeq

from .util import WalkerModel
from .word2vec import Word2VecParameters
from ..typing import Walks
from ..walker import StandardRandomWalker, WalkerParameters

__all__ = [
    'run_gat2vec_unsupervised',
    'Gat2VecUnsupervisedModel',
]


def run_gat2vec_unsupervised(
        graph: Graph,
        structural_vertices: VertexSeq,
        random_walk_parameters: Optional[WalkerParameters] = None,
        word2vec_parameters: Optional[Word2VecParameters] = None
) -> Word2Vec:
    """Run the unsupervised GAT2VEC algorithm to generate a Word2Vec model."""
    model = Gat2VecUnsupervisedModel(
        structural_vertices=structural_vertices,
        random_walk_parameters=random_walk_parameters,
        word2vec_parameters=word2vec_parameters,
    )
    return model.fit(graph)


class Gat2VecUnsupervisedModel(WalkerModel):
    """An implementation of the GAT2VEC unsupervised model [sheikh2018]_ .

    .. [sheikh2018] Sheikh, N., Kefato, Z., & Montresor, A. (2018). Gat2Vec: Representation Learning for Attributed
                    Graphs. Computing, 1–23. https://doi.org/10.1007/s00607-018-0622-9

    .. seealso:: Other Python implementations of GAT2VEC:

        - https://github.com/snash4/GAT2VEC (reference implementation)
    """

    walker_cls = StandardRandomWalker

    def __init__(
            self,
            structural_vertices: Union[Iterable[str], VertexSeq],
            random_walk_parameters: Optional[WalkerParameters] = None,
            word2vec_parameters: Optional[Word2VecParameters] = None
    ) -> None:
        """Initialize the GAT2VEC unsupervised model."""
        super().__init__(
            walker_parameters=random_walk_parameters,
            word2vec_parameters=word2vec_parameters,
        )

        # Double the maximum path length since the paths will be filtered
        self.random_walk_parameters.max_path_length *= 2

        # Store structural vertices - other ones will be filtered out
        if isinstance(structural_vertices, VertexSeq):
            self.structural_vertices = {
                vertex['label']
                for vertex in structural_vertices
            }
        else:
            self.structural_vertices = set(structural_vertices)

    def transform_walks(self, walks: Walks) -> Walks:
        """Remove vertices that aren't labeled as structural vertices from all walks."""
        for walk in walks:
            yield (
                vertex
                for vertex in walk
                if vertex in self.structural_vertices
            )
