from abc import ABC, abstractmethod
import numpy as np
import scipy
import scipy.stats as scs
import networkx as nx
import random
from itertools import combinations, groupby


def gnp_random_connected_graph(n, p, seed):
    random.seed(seed)
    """Generate a random connected graph
    self.num_qubits     : int, number of nodes
    p     : float in [0,1]. Probability of creating an edge
    seed  : int for initialising randomness
    """
    edges = combinations(range(n), 2)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if p <= 0:
        return G
    if p >= 1:
        return nx.complete_graph(n, create_using=G)
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < p:
                G.add_edge(*e)
    return G

class Problem(ABC):
    @abstractmethod
    def generate_qubo():
        pass

class MaxCutNXProblem(Problem):
    def __init__(self, num_qubits, p, seed):
        self.num_qubits = num_qubits
        self.g = gnp_random_connected_graph(num_qubits, p, seed) # Might want to think about other graph generating mechanisms.

    def generate_qubo(self):
        return {(l,r) : 1 for (l,r) in self.g.edges()}


    def graph_features(self):
        """ Big ugly function that computes the features of our networkX graph  """
        num_nodes = len(self.g)
        num_edges = len(self.g.edges)
        density = nx.density(self.g)
        degrees = [d for self.num_qubits,d in self.g.degree()]
        mini, maxi, std, mean, hmean, gmean, skew, kurt = np.min(degrees), np.max(degrees), np.std(degrees), np.mean(degrees), scs.hmean(degrees), scs.gmean(degrees), scs.skew(degrees), scs.kurtosis(degrees)
        avg_shortest_path = nx.average_shortest_path_length(self.g)
        num_triangles = sum(nx.triangles(self.g).values()) / 3
        avg_clustering = nx.average_clustering(self.g)
        diam = nx.diameter(self.g)
        radi = nx.radius(self.g)
        algeb_connec = nx.algebraic_connectivity(self.g)
        laplacian_spectrum = nx.laplacian_spectrum(self.g) # do not include
        laplacian_energy = np.sum(np.abs(laplacian_spectrum-2*num_edges/num_nodes))
        laplacian_matrix = nx.normalized_laplacian_matrix(self.g) # do not include
        norm_laplac_eigvals, _ = scipy.linalg.eig(laplacian_matrix.toarray()) # do not include
        non_zero_eigvals = np.sort(norm_laplac_eigvals[norm_laplac_eigvals != 0]) # do not include
        logratlarge = np.log(non_zero_eigvals[-1]/non_zero_eigvals[-2])
        logratnonzero = np.log(non_zero_eigvals[-1]/non_zero_eigvals[0])
        logratsmall = np.log(non_zero_eigvals[1]/non_zero_eigvals[0])
        return {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "density": density,
                "min": mini,
                "max": maxi,
                "std": std,
                "mean": mean,
                "hmean": hmean,
                "gmean": gmean,
                "skew": skew,
                "kurt": kurt,
                "avg_shortest_path": avg_shortest_path,
                "num_triangles": num_triangles,
                "avg_clustering": avg_clustering,
                "diam": diam,
                "radi": radi,
                "algeb_connec": algeb_connec,
                "laplacian_energy": laplacian_energy,
                "logratlarge": np.real(logratlarge),
                "logratnonzero": np.real(logratnonzero),
                "logratsmall": np.real(logratsmall)
        }



