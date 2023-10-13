import networkx
import numpy as np
import random
import pdb
from latency_predictor.utils import *
from torch_geometric.data import Data
import torch
import pickle

IGNORE_NODE_FEATS = ["Alias", "Filter"]
# IGNORE_NODE_FEATS = ["Alias"]

### TODO: check
## RowsRemovedbyJoinFilter
# MAX_SET_LEN = 50000

HEURISTIC_FEATS = ["heuristic_cost", "heuristic_pred"]
NUM_HIST = 30
STATIC_FEATS=1

class Featurizer():
    def __init__(self,
            plans,
            sys_logs,
            cfg,
            *args,
            **kwargs):
        '''
        '''
        self.kwargs = kwargs
        self.cfg = cfg

        self.num_heuristic_feats = len(HEURISTIC_FEATS)

        for k, val in kwargs.items():
            self.__setattr__(k, val)

        if "ignore_node_feats" in cfg:
            self.ignore_node_feats = cfg["ignore_node_feats"].split(",")
        else:
            self.ignore_node_feats = IGNORE_NODE_FEATS

        print("ignoring feats: ", self.ignore_node_feats)

        if self.cfg["sys_net"]["pretrained_fn"] is not None:
            assert ".wt" in self.cfg["sys_net"]["pretrained_fn"]

        self.normalization_stats = {}
        self.idx_starts = {}
        self.idx_types = {}
        self.val_idxs = {}
        self.idx_lens = {}

        if self.cfg["plan_net"]["arch"] == "onehot":
            # onehot based on qname
            qnames = [g.graph["qname"] for g in plans]
            qnames.sort()
            self.qname_idxs = {}
            for qi,qname in enumerate(qnames):
                self.qname_idxs[qname] = qi
            self.num_features = len(qnames)
            print("Features based on: ", qnames)
        else:
            ## handling plan features / normalizations
            attrs = defaultdict(set)
            self._update_attrs(plans, attrs)

            # if self.y_normalizer != "none":
                # assert False

            # for graph features, each node will reserve spots for each of the
            # features
            self.cur_feature_idx = 0
            self.num_features = 0
            self.num_global_features = 0
            used_keys = set()

            for k,v in attrs.items():
                if k in self.ignore_node_feats:
                    print("ignoring node feature of type: {}, with {} elements".format(k, len(v)))
                    continue

                if self.actual_feats and "Actual" in k:
                    if not "ActualRows" in k:
                        continue
                else:
                    if "Actual" in k:
                        continue

                if len(v) == 1:
                    continue

                assert k not in self.idx_starts
                self.idx_starts[k] = self.cur_feature_idx
                used_keys.add(k)

                v0 = random.sample(v, 1)[0]

                if is_float(v0):
                    # print(k, ": continuous feature")
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

                elif len(v) < self.cfg["max_set_len"]:
                    if not self.feat_onehot:
                        print("skipping features {}, because categorical"\
                                .format(k))
                        del self.idx_starts[k]
                        continue

                    if len(v) < self.cfg["num_bins"]:
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
                        num_vals = self.cfg["num_bins"]
                        self.idx_types[k] = "feature-hashing"

                    self.idx_lens[k] = num_vals
                    self.num_features += num_vals
                    self.cur_feature_idx += num_vals

                else:
                    print("skipping features {}, because too many values: {}"\
                            .format(k, len(v)))
                    # pdb.set_trace()
                    del self.idx_starts[k]

            print("Features based on: ", used_keys)

        if cfg["sys_net"]["save_weights"]:
            pfn = self.cfg["sys_net"]["pretrained_fn"]
            pfn = pfn.replace(".wt", "_graph_normalizers.pkl")
            print("saving graph normalizers to: ", pfn)
            with open(pfn, "wb") as f:
                pickle.dump((self.normalization_stats, self.idx_starts,
                                self.idx_types, self.val_idxs, self.idx_lens,
                                self.num_features),
                            f)

        elif cfg["plan_net"]["pretrained"]:
            pfn = self.cfg["sys_net"]["pretrained_fn"]
            pfn = pfn.replace(".wt", "_graph_normalizers.pkl")
            print("loading graph node normalizers from: " , pfn)

            with open(pfn, "rb") as f:
                    self.normalization_stats, self.idx_starts, \
            self.idx_types, self.val_idxs, self.idx_lens, self.num_features = pickle.load(f)

        if self.y_normalizer == "std":
            yfn = self.cfg["sys_net"]["pretrained_fn"]
            yfn = yfn.replace(".wt", "_ynorms.pkl")

            if cfg["sys_net"]["pretrained"]:
                # load pretrained ys
                with open(yfn, "rb") as f:
                    self.ynorms = pickle.load(f)
                print("Loaded ynormalizers: ", self.ynorms)
            else:
                ys = []
                for G in plans:
                    if self.log_transform_y:
                        ys.append(np.log(G.graph["latency"]))
                    else:
                        ys.append(G.graph["latency"])
                ys = np.array(ys)
                self.ynorms = (np.mean(ys), np.std(ys))
                print("Y normalization with mean/std: ", self.ynorms)
                with open(yfn, "wb") as f:
                    pickle.dump(self.ynorms, f)
        else:
            pass

        ## FIXME: initialize properly
        allfeats = []
        for G in plans:
            allfeats.append(G.graph["runtime_feats"])
        allfeats = np.array(allfeats)
        self.hist_means = np.mean(allfeats,axis=0)
        self.hist_stds = np.std(allfeats,axis=0)

        if "hist_net" not in cfg:
            pass

        elif cfg["hist_net"]["pretrained"]:
            hfn = self.cfg["sys_net"]["pretrained_fn"]
            hfn = hfn.replace(".wt", "_hist_normalizers.pkl")
            with open(hfn, "rb") as f:
                self.hist_means, self.hist_stds = pickle.load(f)
            # for G in plans:
                # G.graph["runtime_feats"] = (G.graph["runtime_feats"] - hist_means) / hist_stds
        else:
            # normalize the 1dfeat reps
            allfeats = []
            for G in plans:
                allfeats.append(G.graph["runtime_feats"])
            allfeats = np.array(allfeats)
            self.hist_means = np.mean(allfeats,axis=0)
            self.hist_stds = np.std(allfeats,axis=0)

            ## just use sys_net params for now
            if cfg["sys_net"]["save_weights"]:
                hfn = self.cfg["sys_net"]["pretrained_fn"]
                hfn = hfn.replace(".wt", "_hist_normalizers.pkl")
                with open(hfn, 'wb') as f:
                    pickle.dump((self.hist_means,self.hist_stds), f)
                print("history normalizers saved at: ", hfn)

            # for G in plans:
                # G.graph["runtime_feats"] = (G.graph["runtime_feats"] - hist_means) / hist_stds

        if self.cfg["sys_net"]["pretrained"]:
            if self.cfg["sys_net"]["use_pretrained_norms"]:
                fn = self.cfg["sys_net"]["pretrained_fn"]
                fn = fn.replace("_fixed", "")
                if ".yaml" in fn:
                    fn = fn.replace(".yaml", "_normalizers.pkl")
                else:
                    fn = fn.replace(".wt", "_normalizers.pkl")

                print("Going to use pretrained log feature normalizers from: ",
                        fn)
                with open(fn, 'rb') as handle:
                    sys_norms = pickle.load(handle)
                self._update_syslog_idx_positions(list(sys_norms.keys()))
            else:
                sys_norms = self._init_syslog_features(sys_logs)
        else:
            sys_norms = self._init_syslog_features(sys_logs)
            if self.cfg["sys_net"]["save_weights"]:
                fn = self.cfg["sys_net"]["pretrained_fn"]

                if ".yaml" in fn:
                    fn = fn.replace(".yaml", "_normalizers.pkl")
                else:
                    fn = fn.replace(".wt", "_normalizers.pkl")

                with open(fn, 'wb') as handle:
                    pickle.dump(sys_norms, handle)
                print("normalizers saved at: ", fn)

        self.sys_norms = sys_norms
        self.normalization_stats.update(sys_norms)

        if self.cfg["factorized_net"]["heuristic_feats"]:
            fn = self.cfg["sys_net"]["pretrained_fn"]

            hfn = fn + "_heuristic_normalizers.pkl"
            if self.cfg["factorized_net"]["pretrained"] and \
                    os.path.exists(hfn):
                print("Going to use pretrained heuristic feature normalizers from: ",
                        hfn)
                with open(hfn, 'rb') as handle:
                    heuristic_norms = pickle.load(handle)
                self.normalization_stats.update(heuristic_norms)
            elif not self.cfg["factorized_net"]["pretrained"]:
                allvals = defaultdict(list)
                for G in plans:
                    for h in HEURISTIC_FEATS:
                        allvals[h].append(G.graph[h])

                heuristic_norms = {}
                for h in HEURISTIC_FEATS:
                    heuristic_norms[h] = (np.mean(allvals[h]), np.std(allvals[h]))

                with open(hfn, 'wb') as handle:
                    pickle.dump(heuristic_norms, handle)
                print("heuristic normalizers saved at: ", hfn)

                self.normalization_stats.update(heuristic_norms)

    def _update_attrs(self, plans, attrs):
        for G in plans:
            for ndata in G.nodes(data=True):
                for key,val in ndata[1].items():
                    # if key in self.ignore_node_feats:
                        # continue
                    attrs[key].add(val)

    def _update_syslog_idx_positions(self, keys):
        keys.sort()
        self.num_syslog_features = 0
        cur_feature_idx = 0

        for key in keys:
            assert key not in self.idx_starts
            self.idx_starts[key] = cur_feature_idx
            self.idx_types[key] = "cont"
            self.idx_lens[key] = 1
            cur_feature_idx += 1
            self.num_syslog_features += 1

        print("Number of system log data points per timestep: ", self.num_syslog_features)
        if STATIC_FEATS:
            self.idx_starts["sys_static"] = self.num_syslog_features
            self.num_syslog_features += 10

    def _init_syslog_features(self, sys_logs):
        print("init syslog features!")
        sys_normalization = {}
        alllogs = []

        for instance in sys_logs:
            for _,curlogs in sys_logs[instance].items():
                curlogs = curlogs.dropna()
                alllogs.append(curlogs)

        df = pd.concat(alllogs)
        df = df.dropna()

        keys = list(df.keys())
        newkeys = []
        for key in keys:
            if not is_float(min(df[key])):
                continue
            if key == "timestamp":
                continue

            assert key not in self.idx_starts
            newkeys.append(key)
        newkeys.sort()
        keys = newkeys

        self._update_syslog_idx_positions(keys)

        for key in keys:
            if self.normalizer == "min-max":
                # TODO: special case w/ unique value
                if max(df[key]) - min(df[key]) == 0:
                    continue
                sys_normalization[key] = (min(df[key]), max(df[key]))

            elif self.normalizer == "std":
                if len(set(df[key])) == 1:
                    # ensures the features just become 0.0; not ignoring this
                    # key to keep consistent #input features for transformers
                    # trained using different logs
                    sys_normalization[key] = (np.mean(df[key].values),
                            1.0)
                else:
                    sys_normalization[key] = (np.mean(df[key].values),
                            np.std(df[key].values))
                    # print(sys_normalization[key])
                    if np.isnan(np.mean(df[key].values)):
                        pdb.set_trace()
            else:
                assert False

        return sys_normalization

    def normalizeY(self, y):
        '''
        TODO.
        '''
        # assert self.y_normalizer == "none"
        if self.log_transform_y:
            np.log(y)

        if self.y_normalizer == "std":
            # val = (prev_logs[key].values.mean() - m0) / m1
            y = (y-self.ynorms[0]) / self.ynorms[1]

        return y

    def unnormalizeY(self, y):
        '''
        TODO.
        '''
        # assert self.y_normalizer == "none"
        if self.log_transform_y:
            np.exp(y)

        if self.y_normalizer == "std":
            y = (y*self.ynorms[1]) + self.ynorms[0]

        return y

    def handle_key(self, key, v, feature):
        if key not in self.idx_starts:
            return
        if key not in self.idx_types:
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

    def get_history_features(self, G, gi, plans):
        hist_feats = []
        for i in range(NUM_HIST):
            # curidx = gi-i-1
            curidx = gi-i
            if curidx < 0:
                curfeats = np.zeros(len(G.graph["runtime_feats"]))
                hist_feats.append(curfeats)
                continue

            curplan = plans[curidx]
            if curplan.graph["instance"] != G.graph["instance"] \
                    or not hasattr(self, "hist_means"):
                curfeats = np.zeros(len(G.graph["runtime_feats"]))
            else:
                curfeats = (np.array(curplan.graph["runtime_feats"]) - self.hist_means) \
                                        / self.hist_stds

            if curidx == gi:
                curfeats *= RUNTIME_MASK

            hist_feats.append(curfeats)

        return torch.tensor(np.array(hist_feats),
                    dtype=torch.float)

    # def get_static_features(self, lt_type):
        # feats = np.zeros(len(ALL_INSTANCES))
        # for i, inst in enumerate(ALL_INSTANCES):
            # if inst == lt_type:
                # feats[i] = 1.0

        # return torch.tensor(feats, dtype=torch.float32)

    def get_log_features(self, prev_logs,
            avg_logs=False, lt_type=""):

        if avg_logs:
            feature = np.zeros(self.num_syslog_features)
        else:
            feature = np.zeros((len(prev_logs), self.num_syslog_features))

        for key in prev_logs.keys():
            if key not in self.idx_starts:
                continue
            idx_col = self.idx_starts[key]
            m0, m1 = self.normalization_stats[key]
            if avg_logs:
                if self.normalizer == "min-max":
                    val = (prev_logs[key].values.mean() - m0) / (m1 - m0)
                elif self.normalizer == "std":
                    val = (prev_logs[key].values.mean() - m0) / m1
                feature[idx_col] = val
            else:
                if self.normalizer == "min-max":
                    try:
                        vals = (prev_logs[key].values - m0) / (m1 - m0)
                    except Exception as e:
                        print(m0, m1, prev_logs[key].values)
                        pdb.set_trace()

                elif self.normalizer == "std":
                    vals = (prev_logs[key].values - m0) / m1

                feature[:,idx_col] = vals

        if "sys_static" in self.idx_starts:
            si = self.idx_starts["sys_static"]
            for i, inst in enumerate(ALL_INSTANCES):
                if inst == lt_type:
                    feature[:,si+i] = 1.0

        feature = torch.tensor(feature, dtype=torch.float)
        return feature

    def get_heuristic_feats(self, G):
        features = np.zeros(self.num_heuristic_feats)
        for hi, hf in enumerate(HEURISTIC_FEATS):
            val = G.graph[hf]

            ## proper normalization
            if hf in self.normalization_stats:
                m0, m1 = self.normalization_stats[hf]
                if m1 == 0:
                    # print("m1 == 0", hf)
                    # pdb.set_trace()
                    m1+=1
                nv = (val-m0) / m1
                if math.isnan(nv):
                    print("nv is nan: ")
                    print(key)
                    print(v)
                    print(nv)
                    pdb.set_trace()
                features[hi] = nv
            else:
                if hi == 0:
                    features[hi] = np.log(val)

        features = torch.tensor(features, dtype=torch.float)
        return features

    def get_pytorch_geometric_features(self, G, subplan_ests=False):
        nodes = list(G.nodes())
        nodes.sort()
        node_dict = {}
        for i, node in enumerate(nodes):
            node_dict[node] = i

        x = []
        for node in nodes:
            x.append(self._featurize_node(G, node))

        edge_idxs = [[],[]]
        for edge in G.edges():
            edge_idxs[0].append(node_dict[edge[1]])
            edge_idxs[1].append(node_dict[edge[0]])
            if self.feat_undirected_edges:
                # put in the other direction as well
                edge_idxs[0].append(node_dict[edge[0]])
                edge_idxs[1].append(node_dict[edge[1]])

        x = np.array(x)
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

