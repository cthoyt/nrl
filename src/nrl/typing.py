# -*- coding: utf-8 -*-

"""Type hints for NRL."""

from typing import Iterable

from igraph import Vertex

__all__ = [
    'Walk',
]

Walk = Iterable[Vertex]
