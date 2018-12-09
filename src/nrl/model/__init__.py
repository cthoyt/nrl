# -*- coding: utf-8 -*-

"""Algorithms for NRL."""

from .deepwalk import DeepWalkModel
from .gat2vec import Gat2VecUnsupervisedModel
from .node2vec import Node2VecModel
from .util import WalkerModel

__all__ = [
    'WalkerModel',
    'DeepWalkModel',
    'Gat2VecUnsupervisedModel',
    'Node2VecModel'
]
