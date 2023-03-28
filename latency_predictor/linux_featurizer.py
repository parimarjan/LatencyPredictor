import networkx
import numpy as np
import random
import pdb
from latency_predictor.utils import *
from torch_geometric.data import Data
import torch
import pickle

PERF_FEATS = ['cycles', 'cycles#', 'instructions', 'instructions#',
        'cache-misses', 'cache-misses#', 'page-faults', 'page-faults#',
        'branch-misses', 'branch-misses#', 'context-switches',
        'context-switches#', 'L1-dcache-load-misses', 'L1-dcache-load-misses#']

class LinuxFeaturizer():
    def __init__(self,
            df,
            sys_logs,
            cfg,
            *args,
            **kwargs):
        '''
        '''
        self.kwargs = kwargs
        self.cfg = cfg

        for k, val in kwargs.items():
            self.__setattr__(k, val)

        self.normalization_stats = {}
        self.idx_starts = {}
        self.idx_types = {}
        self.val_idxs = {}
        self.idx_lens = {}

        ## handling plan features / normalizations
        attrs = defaultdict(set)

        self.num_global_features = 0
        if self.cfg["plan_net"]["feat_type"] == "perf":
            # for graph features, each node will reserve spots for each of the
            # features
            self.cur_feature_idx = 0
            self.num_features = 0
            used_keys = set()
            print("Features based on: ", used_keys)
            # for each job, we will have a unique feature set, based on one
            # instance type
            qnames = list(set(df["qname"]))
            qnames.sort()
            self.qname_features = {}
            defdf = df[df["lt_type"] == "r7g_large_gp2_16g"]
            normalizers = {}

            for key in defdf.keys():
                if key in PERF_FEATS:
                    try:
                        #pdb.set_trace()
                        tmp = defdf[defdf[key].notna()]
                        if len(tmp) == 0:
                            continue
                        if not "float" in str(tmp[key].dtype):
                            tmp = tmp[~tmp[key].str.contains('not', na=False)]
                            if len(tmp) == 0:
                                continue
                        vals = tmp[tmp[key].notna()][key].astype(float)
                    except Exception as e:
                        print(key)
                        print(e)
                        pdb.set_trace()
                        continue
                    if len(vals) == 0:
                        continue
                    normalizers[key] = (np.mean(vals),
                            np.std(vals))

            featkeys = list(normalizers.keys())
            featkeys.sort()

            for qi,qname in enumerate(qnames):
                tmp = defdf[defdf["qname"] == qname]
                feats = np.zeros(len(featkeys))
                for ki, key in enumerate(featkeys):
                    if key not in normalizers:
                        continue
                    tmp = tmp[tmp[key].notna()]
                    if len(tmp) == 0:
                        continue
                    try:
                        val = np.mean(tmp[key])
                    except:
                        continue
                    feats[ki] = (val - normalizers[key][0]) / (normalizers[key][1])

                self.qname_features[qname] = feats

            self.num_features = len(featkeys)
            print("Features based on: ", featkeys)

        elif self.cfg["plan_net"]["feat_type"] == "onehot":
            # onehot based on qname
            qnames = list(set(df["qname"]))
            qnames.sort()
            self.qname_idxs = {}
            for qi,qname in enumerate(qnames):
                self.qname_idxs[qname] = qi
            self.num_features = len(qnames)
            print("Features based on: ", qnames)
        else:
            assert False

        if self.y_normalizer != "none":
            assert False

        if self.cfg["sys_net"]["pretrained"]:
            if self.cfg["sys_net"]["use_pretrained_norms"]:
                fn = self.cfg["sys_net"]["pretrained_fn"]
                fn = fn.replace("_fixed", "")
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
                fn = fn.replace(".wt", "_normalizers.pkl")
                with open(fn, 'wb') as handle:
                    pickle.dump(sys_norms, handle)
                print(fn)

        # for k,v in sys_norms.items():
            # print(k, " mean: ", round(v[0], 2), ", std: ", round(v[1], 2))
        # pdb.set_trace()

        self.normalization_stats.update(sys_norms)

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
            self.idx_starts[key] = cur_feature_idx
            self.idx_types[key] = "cont"
            self.idx_lens[key] = 1
            cur_feature_idx += 1
            self.num_syslog_features += 1

        print("Number of system log data points per timestep: ", self.num_syslog_features)

    def _init_syslog_features(self, sys_logs):

        sys_normalization = {}
        alllogs = []
        for instance in sys_logs:
            for _,curlogs in sys_logs[instance].items():
                alllogs.append(curlogs)

        df = pd.concat(alllogs)

        keys = list(df.keys())
        newkeys = []
        for key in keys:
            if not is_float(min(df[key])):
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
            else:
                assert False

        return sys_normalization

    def normalizeY(self, y):
        '''
        TODO.
        '''
        assert self.y_normalizer == "none"
        if self.log_transform_y:
            np.log(y)
        return y

    def unnormalizeY(self, y):
        '''
        TODO.
        '''
        assert self.y_normalizer == "none"
        if self.log_transform_y:
            np.exp(y)
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

    def get_linux_feats(self, row):
        if self.cfg["plan_net"]["feat_type"] == "onehot":
            feats = np.zeros(self.num_features)
            idx = self.qname_idxs[row["qname"]]
            feats[idx] = 1.0
        else:
            feats = self.qname_features[row["qname"]]

        feats = torch.tensor(feats, dtype=torch.float)
        return feats

    def get_log_features(self, prev_logs, avg_logs=False):

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

        feature = torch.tensor(feature, dtype=torch.float)
        return feature
