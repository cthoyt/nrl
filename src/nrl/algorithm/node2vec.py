import random

import numpy as np


class Node2VecWalker:
    """Create walks using the Node2Vec algorithm for use with Word2Vec."""

    def __init__(self, graph, is_directed, p, q):
        self.graph = graph
        self.is_directed = is_directed
        self.p = p
        self.q = q
        self.alias_nodes, self.alias_edges = self.preprocess_transition_probs(self.graph, self.p, self.q)

    def node2vec_walk(self, walk_length, start):
        """Simulate a random walk starting from start node."""
        walk = [start]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_neighbors = sorted(self.graph.neighbors(cur))
            if 0 == len(cur_neighbors):
                break
            if len(walk) == 1:
                walk.append(cur_neighbors[alias_draw(self.alias_nodes[cur][0], self.alias_nodes[cur][1])])
            else:
                prev = walk[-2]
                next = cur_neighbors[alias_draw(self.alias_edges[prev, cur][0], self.alias_edges[prev, cur][1])]
                walk.append(next)

        return walk

    def simulate_walks(self, num_walks, walk_length):
        """Repeatedly simulate random walks from each node."""
        walks = []
        nodes = list(self.graph.nodes())
        for _ in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walk = self.node2vec_walk(
                    walk_length=walk_length,
                    start=node,
                )
                walks.append(walk)

        return walks

    def preprocess_transition_probs(self, graph, p, q):
        """Preprocessing of transition probabilities for guiding the random walks."""
        is_directed = self.is_directed

        alias_nodes = {}
        for node in graph.nodes():
            unnormalized_probs = [
                graph[node][neighbor]['weight']
                for neighbor in sorted(graph.neighbors(node))
            ]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob) / norm_const
                for u_prob in unnormalized_probs
            ]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}

        if is_directed:
            for source, target in graph.edges():
                alias_edges[source, target] = get_alias_edge(graph, p, q, source, target)
        else:
            for source, target in graph.edges():
                alias_edges[source, target] = get_alias_edge(graph, p, q, source, target)
                alias_edges[target, source] = get_alias_edge(graph, p, q, target, source)

        return alias_nodes, alias_edges


def get_alias_edge(graph, source, target, p, q):
    """Get the alias edge setup lists for a given edge."""
    unnormalized_probs = []

    for target_neighbor in sorted(graph.neighbors(target)):
        if target_neighbor == source:
            unnormalized_probs.append(graph[target][target_neighbor]['weight'] / p)
        elif graph.has_edge(target_neighbor, source):
            unnormalized_probs.append(graph[target][target_neighbor]['weight'])
        else:
            unnormalized_probs.append(graph[target][target_neighbor]['weight'] / q)

    norm_const = sum(unnormalized_probs)
    normalized_probs = [
        float(u_prob) / norm_const
        for u_prob in unnormalized_probs
    ]

    return alias_setup(normalized_probs)


def alias_setup(probs):
    """Compute utility lists for non-uniform sampling from discrete distributions.

    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    """Draw sample from a non-uniform discrete distribution using alias sampling."""
    kk = np.random.randint(len(J))

    if np.random.rand() < q[kk]:
        return kk

    return J[kk]
