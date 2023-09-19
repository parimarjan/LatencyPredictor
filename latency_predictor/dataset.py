import torch
from torch.utils import data
import numpy as np
#from latency_predictor.featurizer import *
from torch_geometric.data import Data, Batch
from networkx.algorithms.traversal.depth_first_search import dfs_tree
import copy
import torch.nn.functional as F

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
            featurizer.num_features, featurizer.num_global_features))

        self.data = self._get_features(plans, syslogs, featurizer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        '''
        '''
        return self.data[index]

    def _get_features(self, plans, sys_logs, featurizer):
        data = []
        skip = 0
        for G in plans:
            if self.subplan_ests:
                assert False

            lat = G.graph["latency"]

            lat = featurizer.normalizeY(lat)
            curx = {}
            curx["y"] = lat

            if featurizer.cfg["plan_net"]["arch"] == "onehot":
                curfeats = featurizer.get_onehot_features(G)
                ## y??
            else:
                curfeats = featurizer.get_pytorch_geometric_features(G,
                        self.subplan_ests)
                # curfeats = curfeats.to(device,
                        # non_blocking=True)

                #curfeats["y"] = lat

            cur_logs = sys_logs[G.graph["tag"]][G.graph["instance"]]

            prev_logs = extract_previous_logs(cur_logs, G.graph["start_time"],
                                    self.log_prev_secs,
                                    self.log_skip,
                                    MAX_LOG_LEN,
                                    )

            if len(prev_logs) == 0:
                skip += 1
                continue

            logf = featurizer.get_log_features(prev_logs,
                                            self.log_avg)
            if logf.shape[0] < MAX_LOG_LEN:
                continue

            # if logf.shape[0] >= 95:
                # # temporary
                # rows = logf.shape[1]
                # to_pad = MAX_LOG_LEN-rows
                # S.append(torch.nn.functional.pad(logf,(0,0,0,to_pad),
                        # mode="constant",value=0))
            # else:
                # rows = x.shape[0]
                # to_pad = MAX_LOG_LEN-rows
                # if to_pad < 0:
                    # S.append(x)
                # else:
                    # S.append(torch.nn.functional.pad(x,(0,0,to_pad,0),
                            # mode="constant",value=0))

            if "col" in featurizer.sys_seq_kind:
                logf = logf.T

            # logf = logf.to(device,
                    # non_blocking=True)

            ## syslogs
            # curfeats["sys_logs"] = logf
            # data.append(curfeats)

            curx["graph"] = curfeats
            curx["sys_logs"] = logf

            info = {}
            info["instance"] = G.graph["instance"]
            info["lt_type"] = G.graph["lt_type"]
            info["latency"] = G.graph["latency"]
            info["qname"] = G.graph["qname"]
            info["bk_kind"] = G.graph["bk_kind"]

            curx["info"] = info
            data.append(curx)

        print("Skipped {} plans because sys logs missing".format(skip))
        return data

class LinuxJobDataset(data.Dataset):
    def __init__(self, df,
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

        self.featurizer = featurizer

        print("node feature length: {}, global_features: {}".format(
            self.featurizer.num_features, self.featurizer.num_global_features))

        self.data = self._get_features(df, syslogs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        '''
        '''
        return self.data[index]

    def _get_features(self, df, sys_logs):
        data = []
        skip = 0

        for i,row in df.iterrows():
            lat = row["runtime"]
            lat = self.featurizer.normalizeY(lat)

            curx = {}

            curfeats = self.featurizer.get_linux_feats(row)
            curx["x"] = curfeats
            curx["y"] = lat

            cur_logs = sys_logs[row["tag"]][row["instance"]]

            prev_logs = extract_previous_logs(cur_logs,
                                    row["start_time"],
                                    self.log_prev_secs,
                                    self.log_skip,
                                    MAX_LOG_LEN,
                                    )

            if len(prev_logs) == 0:
                skip += 1
                print(row["start_time"])
                pdb.set_trace()
                continue

            logf = self.featurizer.get_log_features(prev_logs,
                                                    self.log_avg)


            if logf.shape[0] != MAX_LOG_LEN:
                logf = F.pad(input=logf, pad=(0, 0, 0, MAX_LOG_LEN-logf.shape[0]),
                        mode='constant',
                        value=0)

            curx["sys_logs"] = logf

            info = {}
            info["instance"] = row["instance"]
            info["lt_type"] = row["lt_type"]
            # info["latency"] = row["runtime"]
            info["qname"] = row["qname"]
            curx["info"] = info
            data.append(curx)

        print("Skipped {} plans because sys logs missing".format(skip))
        return data


