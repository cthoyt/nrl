# -*- coding: utf-8 -*-

"""Word2Vec utilities."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from dataclasses_json import dataclass_json
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

from ..typing import Walks

__all__ = [
    'Word2VecParameters',
    'save_word2vec',
    'get_cosine_similarity',
    'get_word2vec_from_walks',
]


@dataclass_json
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
    sim = get_cosine_similarity(word2vec)
    return pd.DataFrame(sim, index=word2vec.wv.index2word, columns=word2vec.wv.index2word)


def get_cosine_similarity(word2vec: Word2Vec) -> np.ndarray:
    """Get the cosine similarity matrix from the embedding.

    Warning; might be very big!
    """
    return cosine_similarity(word2vec.wv.vectors)


def get_word2vec_from_walks(
        walks: Walks,
        word2vec_parameters: Optional[Word2VecParameters] = None,
) -> Word2Vec:
    """Train Word2Vec with the given walks."""
    if word2vec_parameters is None:
        word2vec_parameters = Word2VecParameters()

    # the docs lie, it actually needs this data structure
    walks = [list(walk) for walk in walks]

    return Word2Vec(
        sentences=walks,
        size=word2vec_parameters.size,
        window=word2vec_parameters.window,
        min_count=word2vec_parameters.min_count,
        sg=word2vec_parameters.sg,
        hs=word2vec_parameters.hs,
        workers=word2vec_parameters.workers,
    )
