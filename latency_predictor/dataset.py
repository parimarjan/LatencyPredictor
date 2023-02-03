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
            sys_log_feats,
            subplan_ests=False,
            ):

        for k, val in sys_log_feats.items():
            self.__setattr__(k, val)

        self.subplan_ests = subplan_ests
        self.featurizer = featurizer

        print("node feature length: {}, global_features: {}".format(
            self.featurizer.num_features, self.featurizer.num_global_features))

        self.data = self._get_features(plans, syslogs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        '''
        '''
        return self.data[index]

    def _get_features(self, plans, sys_logs):
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
            cur_logs = sys_logs[G.graph["tag"]]
            prev_logs = extract_previous_logs(cur_logs, G.graph["start_time"],
                                    prev_secs=self.log_prev_secs,
                                    skip_logs = self.log_skip,
                                    )

            logf = self.featurizer.get_log_features(prev_logs,
                                                    self.log_avg)

            ## syslogs
            curfeats["sys_logs"] = logf
            data.append(curfeats)

        return data


