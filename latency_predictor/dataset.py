import torch
from torch.utils import data
import numpy as np
from latency_predictor.featurizer import *
from torch_geometric.data import Data, Batch
from networkx.algorithms.traversal.depth_first_search import dfs_tree
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MAX_LOG_LEN=30

class QueryPlanDataset(data.Dataset):
    def __init__(self, plans,
            syslogs,
            featurizer,
            sys_log_feats,
            subplan_ests=False,
            ):
        for k, val in sys_log_feats.items():
            self.__setattr__(k, val)

        if sys_log_feats["arch"] == "mlp":
            self.log_avg = True
        else:
            self.log_avg = False

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
        skip = 0
        for G in plans:
            if self.subplan_ests:
                assert False

            lat = G.graph["latency"]

            lat = self.featurizer.normalizeY(lat)
            curx = {}

            curfeats = self.featurizer.get_pytorch_geometric_features(G,
                    self.subplan_ests)

            curfeats["y"] = lat

            cur_logs = sys_logs[G.graph["tag"]][G.graph["instance"]]

            prev_logs = extract_previous_logs(cur_logs, G.graph["start_time"],
                                    self.log_prev_secs,
                                    self.log_skip,
                                    MAX_LOG_LEN,
                                    )

            if len(prev_logs) == 0:
                skip += 1
                continue

            logf = self.featurizer.get_log_features(prev_logs,
                                                    self.log_avg)

            ## syslogs
            # curfeats["sys_logs"] = logf
            # data.append(curfeats)

            curx["graph"] = curfeats
            curx["sys_logs"] = logf

            info = {}
            info["instance"] = G.graph["instance"]
            info["latency"] = G.graph["latency"]
            info["qname"] = G.graph["qname"]
            curx["info"] = info

            data.append(curx)

        print("Skipped {} plans because sys logs missing".format(skip))
        return data


