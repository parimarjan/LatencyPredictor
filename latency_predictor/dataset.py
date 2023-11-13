import torch
from torch.utils import data
import numpy as np
from latency_predictor.featurizer import *
from torch_geometric.data import Data, Batch
from networkx.algorithms.traversal.depth_first_search import dfs_tree
import copy
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MAX_LOG_LEN=30

class LatencyConverterDataset(data.Dataset):
    def __init__(self, plans,
            train_df,
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

        self.data = self._get_pair_features(plans, train_df, syslogs,
                featurizer)

        # print(len(self.data))
        # pdb.set_trace()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        '''
        '''
        return self.data[index]

    def _get_pair_features(self, plans, df,
            sys_logs, featurizer):
        start = time.time()
        data = []
        skip = 0

        print("starting iterating over plans")
        for pi, plan1 in enumerate(plans):
            # if pi % 10 == 0:
                # print(pi)

            lat1 = plan1.graph["latency"]
            lat1 = featurizer.normalizeY(lat1)

            cur_logs = sys_logs[plan1.graph["tag"]][plan1.graph["instance"]]

            prev_logs = extract_previous_logs(cur_logs, plan1.graph["start_time"],
                                    self.log_prev_secs,
                                    self.log_skip,
                                    MAX_LOG_LEN,
                                    )

            if len(prev_logs) == 0:
                skip += 1
                continue
            logf1 = featurizer.get_log_features(prev_logs,
                                            self.log_avg)
            tmp = df[(df["qname"] == plan1.graph["qname"])
                    & (df["instance"] != plan1.graph["instance"])
                    ]
            tmp = tmp.sample(frac=0.2)

            for ri, row in tmp.iterrows():
                cur_logs2 = sys_logs[row["tag"]][row["instance"]]

                prev_logs2 = extract_previous_logs(cur_logs2,
                                        row["start_time"],
                                        self.log_prev_secs,
                                        self.log_skip,
                                        MAX_LOG_LEN,
                                        )

                if len(prev_logs2) == 0:
                    skip += 1
                    continue

                logf2 = featurizer.get_log_features(prev_logs2,
                                                self.log_avg)

                lat2 = row["runtime"]
                lat2 = float(featurizer.normalizeY(lat2))

                curx = {}
                curx["y"]  = lat2

                rfeats = plan1.graph["runtime_feats"]
                rfeats = (np.array(rfeats) - featurizer.hist_means) \
                                        / featurizer.hist_stds
                ### previous code:
                # rfeats = torch.tensor(np.tile(rfeats, (30, 1)),
                        # dtype=torch.float)
                # curx["x"] = torch.concatenate([rfeats, logf1, logf2],
                        # axis=1)

                # Duplicate rfeats 30 times to make its shape (30, 10)
                rfeats = torch.tensor(rfeats, dtype=torch.float32)
                rfeats_stacked = rfeats.repeat(30, 1)

                # For the next 30 rows, create a (30, 10) tensor filled with SENTINEL value
                SENTINEL = 0  # Or whatever value you want
                rfeats_sentinel = torch.full((30, rfeats.shape[0]), SENTINEL)

                # Concatenate rfeats and logf1
                rfeats_logf1_combined = torch.cat((rfeats_stacked, logf1), dim=1)

                # Concatenate sentinel value and logf2
                rfeats_sentinel_logf2_combined = torch.cat((rfeats_sentinel, logf2),
                        dim=1)

                # Create a special row of shape (1, 104) filled with the MASKING_TOKEN
                MASKING_TOKEN = 0  # You can set your desired value here
                num_cols = rfeats_logf1_combined.shape[1]
                special_row = torch.full((1, num_cols), MASKING_TOKEN)

                # Use torch.cat to stack tensors with the special row in between
                result = torch.cat((rfeats_logf1_combined, special_row,
                    rfeats_sentinel_logf2_combined), dim=0)
                curx["x"] = result

                data.append(curx)
                if (len(data)) % 2000 == 0:
                    print(len(data))

        print("get pair features took: ", time.time()-start)
        print(len(data))
        return data

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

        for gi, G in enumerate(plans):
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
                                            avg_logs=self.log_avg,
                                            lt_type=G.graph["lt_type"],
                                            )

            # if STATIC_FEATS:
                # lt_feats = featurizer.get_static_features(G.graph["lt_type"])
                # # Duplicate rfeats 30 times to make its shape (30, 10)
                # lt_feats = lt_feats.unsqueeze(0).repeat(30, 1)
                # logf = torch.cat((logf, lt_feats), dim=1)

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
                one_hot = torch.eye(logf.shape[0])
                # Appending the one-hot matrix to the transposed tensor
                logf = torch.cat((logf, one_hot), 1)

            # logf = logf.to(device,
                    # non_blocking=True)

            ## syslogs
            # curfeats["sys_logs"] = logf
            # data.append(curfeats)

            curx["graph"] = curfeats
            curx["sys_logs"] = logf

            hfeats = featurizer.get_heuristic_feats(G)
            curx["heuristic_feats"] = hfeats

            hist = featurizer.get_history_features(G, gi, plans)
            curx["history"] = hist

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


