# -*- coding: utf-8 -*-

"""Input/Output (I/O) utilities for NRL."""

import igraph
from igraph import Graph

__all__ = [
    'read_ncol_graph',
]


def read_ncol_graph(path: str) -> Graph:
    """Read a graph."""
    return igraph.read(path, format='ncol', directed=False, names=True)
