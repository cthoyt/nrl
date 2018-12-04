# -*- coding: utf-8 -*-

"""Algorithms for generating random walks for Node2vec."""

from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
from gensim.models import Word2Vec
from igraph import Graph
from joblib import Parallel, delayed

from nrl.algorithm.random_walk import RandomWalkParameters
from nrl.algorithm.word2vec import Word2VecParameters
from nrl.io import read_ncol_graph

__all__ = [
    'run_node2vec',
]

WEIGHT = 'weight'

AliasData = Tuple[np.ndarray, np.ndarray]


def run_node2vec(graph: Graph,
                 random_walk_parameters: Optional[RandomWalkParameters] = None,
                 word2vec_parameters: Optional[Word2VecParameters] = None) -> Word2Vec:
    """Run node2vec."""
    # walker = Node2VecWalker(graph=graph, is_directed=False, p=1.0, q=1.0)
    # walks = walker.simulate_walks(
    #     walk_length=random_walk_parameters.max_path_length,
    #     num_walks=random_walk_parameters.number_paths,
    # )
    #
    # return get_word2vec_from_walks(
    #     walks,
    #     word2vec_parameters=word2vec_parameters
    # )


class Node2Vec_temp:
    """Implementation of Node2Vec with igraph."""

    FIRST_TRAVEL_KEY = 'first_travel_key'
    PROBS_KEY = 'probabilities'
    NEIGHBORS_KEY = 'neighbors'
    WEIGHT_KEY = 'weight'
    NUM_WALKS_KEY = 'num_walks'
    WALK_LENGTH_KEY = 'walk_length'
    P_KEY = 'p'
    Q_KEY = 'q'

    def __init__(self,
                 graph: Graph,
                 dimensions: int = 128,
                 walk_length: int = 80,
                 num_walks: int = 10,
                 p: float = 1.0,
                 q: float = 1.0,
                 weight_key: str = 'weight',
                 workers: int = 1,
                 sampling_strategy=None,
                 quiet=False) -> None:
        """Initiate the Node2Vec object, precompute walking probabilities, and generate the walks.

        :param graph: Input graph
        :param dimensions: Embedding dimensions (default: 128)
        :param walk_length: Number of nodes in each walk (default: 80)
        :param num_walks: Number of walks per node (default: 10)
        :param p: Return hyper parameter (default: 1)
        :param q: Inout parameter (default: 1)
        :param weight_key: On weighted graphs, this is the key for the weight attribute (default: 'weight')
        :param workers: Number of workers for parallel execution (default: 1)
        :param sampling_strategy: Node specific sampling strategies, supports setting node specific 'q', 'p',
         'num_walks' and 'walk_length'.

        Use these keys exactly. If not set, will use the global ones which were passed on the object initialization
        """
        self.graph = graph
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.weight_key = weight_key
        self.workers = workers
        self.quiet = quiet

        if sampling_strategy is None:
            self.sampling_strategy = {}
        else:
            self.sampling_strategy = sampling_strategy

        self.d_graph = self._precompute_probabilities()
        self.walks = self._generate_walks()

    def _precompute_probabilities(self):
        """Pre-compute transition probabilities for each node."""
        d_graph = defaultdict(dict)
        first_travel_done = set()

        for source in self.graph.vs:
            # Init probabilities dict for first travel
            if self.PROBS_KEY not in d_graph[source['name']]:
                d_graph[source['name']][self.PROBS_KEY] = dict()

            for current_node_id in self.graph.neighbors(source):
                current_node = self.graph.vs[current_node_id]
                current_node_name = int(current_node['name'])
                # Init probabilities dict
                if self.PROBS_KEY not in d_graph[current_node_name]:
                    d_graph[current_node_name][self.PROBS_KEY] = dict()

                unnormalized_weights = list()  # TODO: why not a dict?
                first_travel_weights = list()
                d_neighbors = list()

                # Calculate unnormalized weights
                for target_id in self.graph.neighbors(current_node):
                    target = self.graph.vs[target_id]
                    p = self._get_p(current_node_name)
                    q = self._get_q(current_node_name)

                    edge_weight = \
                        self.graph.es.select(_between=([current_node.index], [target.index]))[
                            self.weight_key][0]
                    ss_weight = self._compute_prob(source.index, target.index, p, q, edge_weight)
                    # Assign the unnormalized sampling strategy weight, normalize during random walk
                    unnormalized_weights.append(ss_weight)

                    if current_node_name not in first_travel_done:
                        first_travel_weights.append(edge_weight)
                    d_neighbors.append(int(target['name']))

                # Normalize
                unnormalized_weights = np.array(unnormalized_weights)
                d_graph[current_node_name][self.PROBS_KEY][
                    int(source['name'])] = unnormalized_weights / unnormalized_weights.sum()

                if current_node_name not in first_travel_done:
                    unnormalized_weights = np.array(first_travel_weights)
                    d_graph[current_node_name][
                        self.FIRST_TRAVEL_KEY] = unnormalized_weights / unnormalized_weights.sum()
                    first_travel_done.add(current_node_name)

                # Save neighbors
                d_graph[current_node_name][self.NEIGHBORS_KEY] = d_neighbors

        return d_graph

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

    def _generate_walks(self):
        """Generate the random walks which will be used as the skip-gram input.

        :return: List of walks. Each walk is a list of nodes.
        """
        flatten = lambda l: [item for sublist in l for item in sublist]

        # Split num_walks for each worker
        num_walks_lists = np.array_split(range(self.num_walks), self.workers)

        walk_results = Parallel(n_jobs=self.workers)(delayed(parallel_generate_walks)(
            self.d_graph,
            self.walk_length,
            len(num_walks),
            idx,
            self.sampling_strategy,
            self.NUM_WALKS_KEY,
            self.WALK_LENGTH_KEY,
            self.NEIGHBORS_KEY,
            self.PROBS_KEY,
            self.FIRST_TRAVEL_KEY,
            self.quiet
        ) for idx, num_walks in enumerate(num_walks_lists, 1))

        walks = flatten(walk_results)

        return walks

    def fit(self, **skip_gram_params) -> Word2Vec:
        """Create the embeddings using gensim's Word2Vec.

        :param skip_gram_params: Parameteres for gensim.models.Word2Vec - do not supply 'size' it is taken from the
         Node2Vec 'dimensions' parameter
        :return: A gensim word2vec model
        """
        if 'workers' not in skip_gram_params:
            skip_gram_params['workers'] = self.workers

        if 'size' not in skip_gram_params:
            skip_gram_params['size'] = self.dimensions

        return Word2Vec(self.walks, **skip_gram_params)


# class Node2VecWalker:
#     """Create walks using the Node2Vec algorithm for use with Word2Vec."""
#
#     def __init__(self,
#                  graph: Graph,
#                  is_directed: bool = False,
#                  p: float = 1.0,
#                  q: float = 1.0) -> None:
#         """Initialize the walker.
#
#         :param graph:
#         :param is_directed:
#         :param p: The return hyper-parameter
#         :param q: The input parameter
#         """
#         self.graph = graph
#         self.is_directed = is_directed
#         self.p = p
#         self.q = q
#
#         self.alias_nodes: Mapping[int, AliasData] = self._get_alias_nodes(graph)
#         self.alias_edges: Mapping[Tuple[int, int], AliasData] = (
#             self._get_directed_alias_edges(graph, p, q)
#             if is_directed else
#             self._get_undirected_alias_edges(graph, p, q)
#         )
#
#     def simulate_walks(self, num_walks: int, walk_length: int):
#         """Repeatedly simulate random walks from each node."""
#         walks = []
#         vertices = list(self.graph.vs)
#         for _ in range(num_walks):
#             random.shuffle(vertices)
#             for vertex in vertices:
#                 walk = self.node2vec_walk(
#                     walk_length=walk_length,
#                     start=vertex.index,
#                 )
#                 walks.append(walk)
#
#         return walks
#
#     def node2vec_walk(self, walk_length: int, start: int) -> List[int]:
#         """Simulate a random walk starting from start node."""
#         walk: List[int] = [start]
#
#         if 0 == self.graph.neighborhood_size(start):
#             return walk  # this start node has no neighbors
#
#         j, q = self.alias_nodes[start]
#         print()
#         print('J', j)
#         print('q', q)
#
#         start_neighbor_idx = alias_draw(j, q)
#         print('start index', start_neighbor_idx)
#
#         start_neighbors = sorted(self.graph.neighbors(start))
#         print('start neighbors', start_neighbors)
#         walk.append(start_neighbors[start_neighbor_idx])
#
#         while len(walk) < walk_length:
#             current_vertex = walk[-1]
#
#             if 0 == self.graph.neighborhood_size(current_vertex):
#                 break
#
#             prev = walk[-2]
#             j, q = self.alias_edges[prev, current_vertex]
#             idx = alias_draw(j, q)
#
#             cur_neighbors = sorted(self.graph.neighbors(current_vertex))
#             walk.append(cur_neighbors[idx])
#
#         return walk
#
#     @staticmethod
#     def _get_alias_nodes(graph: Graph) -> Mapping[int, AliasData]:
#         return {
#             vertex.index: alias_setup(
#                 graph.es.select(_source=vertex)[WEIGHT]
#             )
#             for vertex in graph.vs
#         }
#
#     @staticmethod
#     def _get_directed_alias_edges(graph: Graph, p: float, q: float) -> Mapping[
#         Tuple[int, int], AliasData]:
#         return {
#             (source, target): get_alias_edge(graph, source, target, p, q)
#             for source, target in graph.es
#         }
#
#     @staticmethod
#     def _get_undirected_alias_edges(graph: Graph, p: float, q: float) -> Mapping[
#         Tuple[int, int], AliasData]:
#         alias_edges = {}
#
#         for edge in graph.es:
#             source, target = edge.source, edge.target
#             alias_edges[source, target] = get_alias_edge(graph, source, target, p, q)
#             alias_edges[target, source] = get_alias_edge(graph, target, source, p, q)
#
#         return alias_edges
#
#
# def get_alias_edge(graph: Graph, source: Vertex, target: Vertex, p: float, q: float):
#     """Get the alias edge setup lists for a given edge."""
#     probs = []
#
#     for target_neighbor in sorted(graph.neighbors(target)):
#         weight = graph.es.find(_source=target, _target=target_neighbor)[WEIGHT]
#
#         if target_neighbor == source:
#             prob = weight / p
#         elif graph.are_connected(target_neighbor, source):
#             prob = weight
#         else:
#             prob = weight / q
#
#         probs.append(prob)
#
#     return alias_setup(probs)
#
#
# def normalize(numbers: List[float]) -> List[float]:
#     total = sum(numbers)
#     return [number / total for number in numbers]
#
#
# def alias_setup(probs: List[float]) -> AliasData:
#     """Compute utility lists for non-uniform sampling from discrete distributions.
#
#     Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
#     for details
#     """
#     probs = normalize(probs)
#     K = len(probs)
#     q = np.zeros(K)
#     j = np.zeros(K, dtype=np.int)
#
#     smaller = []
#     larger = []
#     for kk, prob in enumerate(probs):
#         q[kk] = K * prob
#         if q[kk] < 1.0:
#             smaller.append(kk)
#         else:
#             larger.append(kk)
#
#     while len(smaller) > 0 and len(larger) > 0:
#         small = smaller.pop()
#         large = larger.pop()
#
#         j[small] = large
#         q[large] = q[large] + q[small] - 1.0
#         if q[large] < 1.0:
#             smaller.append(large)
#         else:
#             larger.append(large)
#
#     return j, q
#
#
# def alias_draw(j, q):
#     """Draw sample from a non-uniform discrete distribution using alias sampling."""
#     kk = np.random.randint(len(j))
#
#     if np.random.rand() < q[kk]:
#         return kk
#
#     return j[kk]
if __name__ == '__main__':
    graph = read_ncol_graph('/home/omuslu/Documents/nrl/tests/resources/weighted_network.edgelist')
    nrl_n2v = Node2Vec_temp(graph)
    nrl_probs_dict = nrl_n2v._precompute_probabilities()
