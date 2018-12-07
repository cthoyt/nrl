# -*- coding: utf-8 -*-

"""Input/Output (I/O) utilities for networks in NDEx."""

import ndex2
from igraph import Graph
from tqdm import tqdm

__all__ = ['get_ndex_graph']


def get_ndex_graph(network_id: str) -> Graph:
    """Get a graph from NDEx"""
    ndex_client = ndex2.client.Ndex2()

    res = ndex_client.get_network_as_cx_stream(network_id)
    res_json = res.json()

    node_data = {}
    node_labels = {}
    nodes_added = set()

    graph = Graph()

    # FIXME
    for entry in tqdm(res_json, desc='entry', leave=True):
        for aspect, data in tqdm(entry.items(), desc='aspect', leave=False):
            if aspect == 'nodes':
                for node in tqdm(data, desc='nodes', leave=False):
                    node_data[str(node['@id'])] = node
                    node_labels[str(node['@id'])] = node['n']

            if aspect == 'edges':
                for edge in tqdm(data, desc='edges', leave=False):
                    nodes_added.add(edge['s'])
                    nodes_added.add(edge['t'])
                    G[str(edge['s'])].append(str(edge['t']))

    return graph
