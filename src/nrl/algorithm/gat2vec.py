# -*- coding: utf-8 -*-

"""An implementation of the GAT2VEC extension of the DeepWalk algorithm."""

from typing import Iterable, Optional

from gensim.models import Word2Vec
from igraph import Graph, VertexSeq

from .util import BaseModel
from .word2vec import Word2VecParameters, get_word2vec_from_walks
from ..walker import RandomWalkParameters, StandardRandomWalker, Walk

__all__ = [
    'run_gat2vec_unsupervised',
    'Gat2VecUnsupervisedModel',
]


def run_gat2vec_unsupervised(graph: Graph,
                             structural_vertices: VertexSeq,
                             random_walk_parameters: Optional[RandomWalkParameters] = None,
                             word2vec_parameters: Optional[Word2VecParameters] = None
                             ) -> Word2Vec:
    """Run the unsupervised GAT2VEC algorithm.

    :param graph:
    :param structural_vertices:
    :param random_walk_parameters:
    :param word2vec_parameters:
    """
    model = Gat2VecUnsupervisedModel(
        graph=graph,
        structural_vertices=structural_vertices,
        random_walk_parameters=random_walk_parameters,
        word2vec_parameters=word2vec_parameters,
    )

    return model.fit()


class Gat2VecUnsupervisedModel(BaseModel):
    """An implementation of the GAT2VEC [1]_ model.

    .. [1] Sheikh, N., Kefato, Z., & Montresor, A. (2018). Gat2Vec: Representation Learning for Attributed Graphs.
           Computing, 1–23. https://doi.org/10.1007/s00607-018-0622-9

    .. seealso:: Other Python implementations of GAT2VEC:

        - https://github.com/snash4/GAT2VEC (reference implementation)
    """

    def __init__(self,
                 graph: Graph,
                 structural_vertices: VertexSeq,
                 random_walk_parameters: Optional[RandomWalkParameters] = None,
                 word2vec_parameters: Optional[Word2VecParameters] = None
                 ) -> None:
        """Initialize the GAT2VEC model."""
        super().__init__(
            graph=graph,
            random_walk_parameters=random_walk_parameters,
            word2vec_parameters=word2vec_parameters,
        )

        # Double the maximium path length since the paths will be filtered
        self.random_walk_parameters.max_path_length *= 2

        # Store structural verticies - other ones will be filtered out
        self.structural_vertices = structural_vertices

    def fit(self):
        """Fit the GAT2VEC model to the graph with the given parameters."""
        walker = StandardRandomWalker(self.random_walk_parameters)
        walks = walker.get_walks(self.graph)

        # stringify output from igraph for Word2Vec
        # filter out vertices that aren't structural vertices
        walks = self._transform_walks(walks)

        return get_word2vec_from_walks(
            walks=walks,
            word2vec_parameters=self.word2vec_parameters,
        )

    def _transform_walks(self, walks: Iterable[Walk]) -> Iterable[Iterable[str]]:
        for walk in walks:
            yield (
                str(vertex)
                for vertex in walk
                if vertex in self.structural_vertices
            )
