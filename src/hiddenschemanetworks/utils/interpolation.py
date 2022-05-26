import math

import networkx as nx
import numpy as np
import torch
from random import random


class Interpolation:
    '''Represents an interpolation between two walks in a graph of equal length'''

    def __init__(self, g: torch.Tensor, paths: torch.Tensor, w=None):
        '''
        Init an Interpolation Object that holds information on shortest paths between corresponding vertices in two random walks.

        g: Graph object
        p1, p2: lists of vertices in g that represent walks
        w: the edge attribute that holds length information (if None, edge weights are assumed to be 1)
        '''

        self.p1 = self.to_list(paths[0, :, :])
        self.p2 = self.to_list(paths[1, :, :])
        self.g = nx.from_numpy_array(g.numpy())
        self.w = w
        self.interpolation_paths = list()
        for i, p in enumerate(self.p1):
            self.interpolation_paths.append(nx.shortest_path(
                self.g, source=p, target=self.p2[i], weight=w))

    def to_list(self, tensor):
        return np.nonzero(tensor.numpy())[1]

    def shortest_path_lengths(self):
        '''Return the lenghts of all shortest paths, in sequence'''
        return [len(p) for p in self.interpolation_paths]

    def interpolated_walk(self, pam: float = 0.5):
        '''Return the pam interpolation between p1 and p2. That is, return the floor(pam*len(p)) th element from the shortest path p for all indices i'''
        idxs = [math.floor(pam * len(p)) for p in self.interpolation_paths]
        return [p[i] for p, i in zip(self.interpolation_paths, idxs)]

    def random_interpolated_walk(self):
        '''Return a random interpolation between p1 and p2. that is, for each index i, return a random element from the ith path'''
        idxs = [random.randint(0, len(p)-1) for p in self.interpolation_paths]
        return [p[i] for p, i in zip(self.interpolation_paths, idxs)]

    def single_position_interpolated_walk(self, i: int = 0, pam: float = 0.5):
        '''Return an interpolation path that changes only vertex of position i and otherwise returns path1'''
        idx = math.floor(pam * len(self.interpolation_paths[i]))
        p = [x for x in self.p1]
        p[i] = self.interpolation_paths[i][idx]
        return p

    def recombine_at_position(self, i: int = 0):
        '''Return concatenation of p1[:i] and p2[i:]'''
        return np.concatenate((self.p1[:i], self.p2[i:]), axis=0)

    def to_onehot(self, path):
        path2 = np.zeros([len(path), self.g.number_of_nodes()])
        path2[range(len(path)), path] = 1
        return path2
