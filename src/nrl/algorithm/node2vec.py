# -*- coding: utf-8 -*-

"""Algorithms for generating random walks for Node2vec."""

from typing import Optional

import numpy as np
from gensim.models import Word2Vec
from igraph import Graph

from .util import BaseModel
from .word2vec import Word2VecParameters, get_word2vec_from_walks
from ..walker import BiasedRandomWalker, RandomWalkParameters

__all__ = [
    'Node2VecModel',
]

WEIGHT = 'weight'


class Node2VecModel(BaseModel):
    """An implementation of the Node2Vec [1]_ model.

    .. [1] Grover, A., & Leskovec, J. (2016). Node2Vec: Scalable Feature Learning for Networks. In Proceedings of the
           22Nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 855–864). New York, NY,
           USA: ACM. https://doi.org/10.1145/2939672.2939754

    .. seealso:: Other Python implementations of Node2Vec:

        - https://github.com/aditya-grover/node2vec (reference implementation)
        - https://github.com/eliorc/node2vec (this is what you get with `pip install node2vec`)
        - https://github.com/thunlp/OpenNE
        - https://github.com/apple2373/node2vec
    """

    FIRST_TRAVEL_KEY = 'first_travel_key'
    PROBS_KEY = 'probabilities'
    WEIGHT_KEY = 'weight'
    NUM_WALKS_KEY = 'num_walks'
    WALK_LENGTH_KEY = 'walk_length'
    P_KEY = 'p'
    Q_KEY = 'q'

    def __init__(self,
                 graph: Graph,
                 random_walk_parameters: Optional[RandomWalkParameters] = None,
                 word2vec_parameters: Optional[Word2VecParameters] = None
                 ) -> None:
        """Initialize the Node2Vec model."""
        super().__init__(
            graph=graph,
            random_walk_parameters=random_walk_parameters,
            word2vec_parameters=word2vec_parameters
        )

        self.dimensions = word2vec_parameters.size
        self.walk_length = random_walk_parameters.max_path_length
        self.num_walks = random_walk_parameters.number_paths
        self.p = random_walk_parameters.p
        self.q = random_walk_parameters.q
        self.weight_key = WEIGHT
        self.workers = word2vec_parameters.workers

        sampling_strategy = random_walk_parameters.sampling_strategy
        if sampling_strategy is not None:
            self.sampling_strategy = sampling_strategy

        if not random_walk_parameters.is_weighted:
            for edge in self.graph.es:
                edge['weight'] = 1.0

        self._precompute_probs()

    def _precompute_probs(self):
        """Pre-compute transition probabilities for each node."""
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
            edge_weight = edge[self.weight_key][0]

            # Assign the unnormalized sampling strategy weight, normalize during random walk
            unnormalized_weights.append(
                self._compute_prob(source.index, target.index, p, q, edge_weight)
            )

            if current_node['name'] not in first_travel_done:
                first_travel_weights.append(edge_weight)

        return unnormalized_weights, first_travel_weights

    def _get_p(self, current_node):
        p = self.sampling_strategy[current_node].get(
            self.P_KEY,
            self.p
        ) if current_node in self.sampling_strategy else self.p
        return p

    def _get_q(self, current_node):
        q = self.sampling_strategy[current_node].get(
            self.Q_KEY,
            self.q
        ) if current_node in self.sampling_strategy else self.q
        return q

    def _compute_prob(self, source, target, p, q, weight):
        if target == source:
            return weight / p
        elif len(self.graph.es.select(_source=source, _target=target)) > 0:
            return weight
        return weight / q

    def fit(self) -> Word2Vec:
        """Create the embeddings using gensim's Word2Vec."""
        walker = BiasedRandomWalker(self.random_walk_parameters)
        walks = walker.get_walks(self.graph)

        # stringify output from igraph for Word2Vec
        walks = self._transform_walks(walks)

        return get_word2vec_from_walks(walks)
