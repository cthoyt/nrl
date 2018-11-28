# -*- coding: utf-8 -*-

"""Word2Vec utilities."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

__all__ = [
    'Word2VecParameters',
    'save_word2vec',
    'get_cosine_similarity_df',
    'get_cosine_similarity',
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


def save_word2vec(word2vec: Word2Vec, name: str):
    """Save the embedding."""
    word2vec.wv.save_word2vec_format(fname=name)


def get_cosine_similarity_df(word2vec: Word2Vec) -> pd.DataFrame:
    """Get the cosine similarity matrix from the embedding as a Pandas DataFrame."""
    node_labels = ...
    labels = [node_labels[n] for n in word2vec.wv.index2word]
    sim = get_cosine_similarity(word2vec)
    return pd.DataFrame(sim, index=labels, columns=labels)


def get_cosine_similarity(word2vec: Word2Vec) -> np.ndarray:
    """Get the cosine similarity matrix from the embedding.

    Warning; might be very big!
    """
    return 1 - cosine_similarity(word2vec.wv.vectors)
