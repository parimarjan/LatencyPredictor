import torch
from torch.utils import data
import numpy as np
from latency_predictor.featurizer import *
from torch_geometric.data import Data, Batch
from networkx.algorithms.traversal.depth_first_search import dfs_tree
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QueryPlanDataset(data.Dataset):
    def __init__(self, plans,
            syslogs,
            featurizer,
            subplan_ests=False
            ):

        self.subplan_ests = subplan_ests
        self.featurizer = featurizer
        # self.num_features = self.featurizer.num_features
        # self.num_global_features = self.featurizer.num_global_features
        print("node feature length: {}, global_features: {}".format(
            self.featurizer.num_features, self.featurizer.num_global_features))

        self.data = self._get_features_geometric(plans, syslogs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        '''
        '''
        return self.data[index]

    def _get_features_geometric(self, plans, syslogs):
        data = []

        for G in plans:
            if self.subplan_ests:
                assert False

            lat = G.graph["latency"]

            lat = self.featurizer.normalizeY(lat)
            curfeats = self.featurizer.get_pytorch_geometric_features(G,
                    self.subplan_ests)

            curfeats["y"] = lat
            info = {}

            ## extra info;
            curfeats["info"] = info

            ## syslogs
            curfeats["syslogs"] = []
            data.append(curfeats)

        return data


