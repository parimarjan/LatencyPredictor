import networkx
import numpy as np
import random
import pdb
from latency_predictor.utils import *
from torch_geometric.data import Data
import torch

IGNORE_NODE_FEATS = ["Alias"]

### TODO: check
## RowsRemovedbyJoinFilter
MAX_SET_LEN = 50

class Featurizer():
    def __init__(self,
            plans,
            *args,
            **kwargs):
        '''
        '''
        self.kwargs = kwargs
        for k, val in kwargs.items():
            self.__setattr__(k, val)
        self.ignore_node_feats = IGNORE_NODE_FEATS

        self.normalization_stats = {}
        attrs = defaultdict(set)

        self._update_attrs(plans, attrs)

        if self.y_normalizer != "none":
            assert False

        self.idx_starts = {}
        self.idx_types = {}
        self.val_idxs = {}
        self.idx_lens = {}

        # for graph features, each node will reserve spots for each of the
        # features
        self.cur_feature_idx = 0
        self.num_features = 0
        self.num_global_features = 0

        # print("Features based on: ", attrs.keys())
        used_keys = set()
        for k,v in attrs.items():
            if k in self.ignore_node_feats:
                print("ignoring node feature of type: {}, with {} elements".format(k, len(v)))
                continue

            if "Actual" in k:
                continue

            if len(v) == 1:
                continue

            print(k, len(v))

            self.idx_starts[k] = self.cur_feature_idx
            used_keys.add(k)

            v0 = random.sample(v, 1)[0]
            if len(v) < MAX_SET_LEN:
                if len(v) < self.num_bins:
                    print(k, ": one-hot using bins")
                    num_vals = len(v)
                    self.idx_types[k] = "one-hot"
                    # each unique value is mapped to an index
                    self.val_idxs[k] = {}
                    v = list(v)
                    v.sort()
                    for i, ve in enumerate(v):
                        self.val_idxs[k][ve] = i
                else:
                    print(k, ": feature-hashing")
                    num_vals = self.num_bins
                    self.idx_types[k] = "feature-hashing"

                self.idx_lens[k] = num_vals
                self.num_features += num_vals
                self.cur_feature_idx += num_vals

            elif is_float(v0):
                print(k, ": continuous feature")
                cvs = [float(v0) for v0 in v]
                if self.normalizer == "min-max":
                    self.normalization_stats[k] = (min(cvs), max(cvs))
                elif self.normalizer == "std":
                    self.normalization_stats[k] = (np.mean(cvs), np.std(cvs))
                else:
                    assert False

                self.idx_types[k] = "cont"
                self.idx_lens[k] = 1
                self.cur_feature_idx += 1
                self.num_features += 1
            else:
                print("skipping features {}, because too many values: {}"\
                        .format(k, len(v)))
                del self.idx_starts[k]

        print("Features based on: ", used_keys)

    def _update_attrs(self, plans, attrs):
        for G in plans:
            for ndata in G.nodes(data=True):
                for key,val in ndata[1].items():
                    # if key in self.ignore_node_feats:
                        # continue
                    attrs[key].add(val)

    def normalizeY(self, y):
        '''
        TODO.
        '''
        assert self.y_normalizer == "none"
        return y

    def unnormalizeY(self, y):
        '''
        TODO.
        '''
        assert self.y_normalizer == "none"
        return y

    def handle_key(self, key, v, feature):
        if key not in self.idx_starts:
            return

        if self.idx_types[key] == "one-hot":
            if v in self.val_idxs[key]:
                idx = self.val_idxs[key][v]
                feature[self.idx_starts[key] + idx] = 1.0
        elif self.idx_types[key] == "one-hot-max":
            if v in self.val_idxs[key]:
                idx = self.val_idxs[key][v]
                feature[self.idx_starts[key] + idx] = 1.0
            else:
                # put it on the max-index
                midx = max(self.val_idxs[key])
                feature[self.idx_starts[key] + midx] = 1.0

        elif self.idx_types[key] == "feature-hashing":
            idx = deterministic_hash(v) % self.idx_lens[key]
            feature[self.idx_starts[key] + idx] = 1.0

        elif self.idx_types[key] == "normalized":
            feature[self.idx_starts[key]] = v

        elif self.idx_types[key] == "cont":
            m0, m1 = self.normalization_stats[key]
            v = float(v)
            if self.normalizer == "min-max":
                if (m1 - m0) == 0:
                    print("m1-m0=0 for: ", key)
                    return
                nv = (v-m0) / (m1 - m0)
            elif self.normalizer == "std":
                if m1 == 0:
                    print("m1 == 0", key)
                    pdb.set_trace()
                    return
                nv = (v-m0) / m1
                if math.isnan(nv):
                    print("nv is nan: ")
                    print(key)
                    print(v)
                    print(nv)
                    pdb.set_trace()
            else:
                assert False

            feature[self.idx_starts[key]] = nv
        else:
            assert False

    def _featurize_node(self, G, node):
        feature = np.zeros(self.num_features)
        for k,v in G.nodes()[node].items():
            self.handle_key(k, v, feature)

        ### additional / derived features

        if self.feat_noncumulative_costs:
            pass

        # if self.feat_size_est:
            # rs = float(G.nodes()[node]["@AvgRowSize"])
            # erows = float(G.nodes()[node]["@EstimateRows"])
            # estsize = rs*erows
            # self.handle_key("SizeEst", erows, feature, wkey)

        # if self.feat_num_nodes:
            # num_nodes = G.graph["num_nodes"]
            # feature[self.idx_starts["NumNodes"]] = num_nodes / self.max_nodes

        if self.feat_subtree_summary:
            from networkx.algorithms.traversal.depth_first_search import dfs_tree
            subtree = dfs_tree(G, node)
            pass

        return feature

    def get_pytorch_geometric_features(self, G, subplan_ests=False):
        nodes = list(G.nodes())
        nodes.sort()
        node_dict = {}
        for i, node in enumerate(nodes):
            node_dict[node] = i

        x = []
        for node in nodes:
            x.append(self._featurize_node(G, node))

        # going to reverse all direction of edges, as costs should be computed
        # going into the Top root node
        edge_idxs = [[],[]]
        for edge in G.edges():
            edge_idxs[0].append(node_dict[edge[1]])
            edge_idxs[1].append(node_dict[edge[0]])
            if self.feat_undirected_edges:
                # put in the other direction as well
                edge_idxs[0].append(node_dict[edge[0]])
                edge_idxs[1].append(node_dict[edge[1]])

        x = torch.tensor(x, dtype=torch.float)
        edge_idxs = torch.tensor(edge_idxs, dtype=torch.long)

        if subplan_ests:
            assert False
            # subp_ests = []
            # for node in nodes:
                # if "StepsTime" not in G.nodes()[node]:
                    # subp_ests.append(0.0)
                # else:
                    # subp_ests.append(G.nodes()[node]["StepsTime"])
            # data = Data(x=x, edge_index=edge_idxs, y=subp_ests)
        else:
            data = Data(x=x, edge_index=edge_idxs)

        return data

