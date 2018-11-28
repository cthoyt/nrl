# -*- coding: utf-8 -*-

"""An implementation of the DeepWalk algorithm."""

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from gensim.models import Word2Vec
from igraph import Graph
from sklearn.metrics.pairwise import cosine_similarity

from .random_walk import get_random_walks

__all__ = [
    'get_word2vec',
    'save_word2vec',
]


@dataclass
class Word2VecParameters:
    """Parameters for :py:class:`gensim.models.Word2Vec`."""

    #: The dimensionality of the embedding
    size: int = 128
    #: The size of the sliding window
    window: int = 5
    #:
    min_count: int = 0
    #:
    sg: int = 1
    #:
    hs: int = 1
    #:
    workers: int = 1


def get_word2vec(graph: Graph,
                 number_paths: int = 10,
                 max_path_length: int = 80,
                 word2vec_parameters: Optional[Word2VecParameters] = None) -> Word2Vec:
    """Build a Word2Vec model using random walks on the graph"""
    walks = get_random_walks(
        graph=graph,
        number_paths=number_paths,
        max_path_length=max_path_length,
    )

    walks = [
        list(map(str, walk))
        for walk in walks
    ]

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


def save_word2vec(word2vec: Word2Vec, name: str):
    """Save the embedding."""
    word2vec.wv.save_word2vec_format(fname=name)


def get_cosine_similarity_df(word2vec: Word2Vec) -> pd.DataFrame:
    """Get the cosine similarity matrix from the embedding as a Pandas DataFrame."""
    node_labels = ...
    labels = [node_labels[n] for n in word2vec.wv.index2word]
    sim = get_cosine_similarity(word2vec)
    return pd.DataFrame(sim, index=labels, columns=labels)


def get_cosine_similarity(word2vec: Word2Vec):
    """Get the cosine similarity matrix from the embedding.

    Warning; might be very big!
    """
    return 1 - cosine_similarity(word2vec.wv.vectors)
