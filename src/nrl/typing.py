# -*- coding: utf-8 -*-

"""Type hints for NRL."""

from typing import Iterable, Union

import igraph
import networkx

__all__ = [
    'Graph',
    'Walk',
    'Walks',
]

Graph = Union[igraph.Graph, networkx.Graph]
Walk = Iterable[str]
Walks = Iterable[Walk]
