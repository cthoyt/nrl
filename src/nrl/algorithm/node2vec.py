# -*- coding: utf-8 -*-

"""Algorithms for generating random walks for Node2vec."""

import numpy as np

from .util import WalkerModel
from ..walker import BiasedRandomWalker

__all__ = [
    'Node2VecModel',
]


# TODO make pre-processing the graph a separate function / class

class Node2VecModel(WalkerModel):
    """An implementation of the Node2Vec [grover2016]_ model.

    .. [grover2016] Grover, A., & Leskovec, J. (2016). Node2Vec: Scalable Feature Learning for Networks. In Proceedings
                    of the 22Nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining
                    (pp. 855–864). New York, NY, USA: ACM. https://doi.org/10.1145/2939672.2939754

    .. seealso:: Other Python implementations of Node2Vec:

        - https://github.com/aditya-grover/node2vec (reference implementation)
        - https://github.com/eliorc/node2vec (this is what you get with `pip install node2vec`)
        - https://github.com/thunlp/OpenNE
        - https://github.com/apple2373/node2vec
    """

    random_walker_cls = BiasedRandomWalker

    NUM_WALKS_KEY = 'num_walks'
    WALK_LENGTH_KEY = 'walk_length'
    PROBS_KEY = 'probabilities'
    FIRST_TRAVEL_KEY = 'first_travel_key'

    WEIGHT_KEY = 'weight'

    P_KEY = 'p'
    Q_KEY = 'q'

    def initialize(self):
        """Pre-process the model by computing transition probabilities for each node in the graph."""
        if not self.random_walk_parameters.is_weighted:
            for edge in self.graph.es:
                edge['weight'] = 1.0

        first_travel_done = set()

        for node in self.graph.vs:
            node[self.PROBS_KEY] = dict()

        for source in self.graph.vs:
            for current_node in source.neighbors():
                unnormalized_weights, first_travel_weights = self._compute_unnormalized_weights(
                    source,
                    current_node,
                    first_travel_done
                )

                # Normalize
                unnormalized_weights = np.array(unnormalized_weights)
                sum_of_weights = unnormalized_weights.sum()
                current_node[self.PROBS_KEY][source['name']] = unnormalized_weights / sum_of_weights

                if current_node['name'] not in first_travel_done:
                    unnormalized_weights = np.array(first_travel_weights)
                    sum_of_weights = unnormalized_weights.sum()
                    current_node[self.FIRST_TRAVEL_KEY] = unnormalized_weights / sum_of_weights
                    first_travel_done.add(current_node['name'])

    def _compute_unnormalized_weights(self, source, current_node, first_travel_done):
        """Compute the unnormalized weights for Node2Vec algorithm.

        :param source: The source node of the previous step on the walk.
        :param current_node: The target node of the previous step on the walk.
        :param first_travel_done: A set
        :return:
        """
        unnormalized_weights = list()  # TODO: why not a dict?
        first_travel_weights = list()

        # Calculate unnormalized weights
        for target in current_node.neighbors():
            p = self._get_p(current_node['name'])
            q = self._get_q(current_node['name'])

            edge = self.graph.es.select(_between=([current_node.index], [target.index]))
            edge_weight = edge[self.WEIGHT_KEY][0]

            # Assign the unnormalized sampling strategy weight, normalize during random walk
            unnormalized_weights.append(
                self._compute_prob(source.index, target.index, p, q, edge_weight)
            )

            if current_node['name'] not in first_travel_done:
                first_travel_weights.append(edge_weight)

        return unnormalized_weights, first_travel_weights

    def _get_p(self, current_node):
        p = self.random_walk_parameters.sampling_strategy[current_node].get(
            self.P_KEY,
            self.random_walk_parameters.p
        ) if current_node in self.random_walk_parameters.sampling_strategy else self.random_walk_parameters.p
        return p

    def _get_q(self, current_node):
        q = self.random_walk_parameters.sampling_strategy[current_node].get(
            self.Q_KEY,
            self.random_walk_parameters.q
        ) if current_node in self.random_walk_parameters.sampling_strategy else self.random_walk_parameters.q
        return q

    def _compute_prob(self, source, target, p, q, weight):
        if target == source:
            return weight / p
        elif len(self.graph.es.select(_source=source, _target=target)) > 0:
            return weight
        return weight / q
