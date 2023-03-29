import torch
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_
import torch.nn.functional as F
import torch_geometric

import numpy as np
import random
import os
import sys

import time
import glob

from latency_predictor.dataset import LinuxJobDataset
from latency_predictor.eval_fns import *
from latency_predictor.utils import *
from latency_predictor.nets import *
from latency_predictor.algs import *

import pickle
import wandb

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def collate_fn_linux(X):
    Z = [x["x"] for x in X]
    S = []
    infos = []

    for curx in X:
        x = curx["sys_logs"]
        infos.append(curx["info"])
        if len(x.shape) == 1:
            S.append(x)
        else:
            rows = x.shape[0]
            to_pad = MAX_LOG_LEN-rows
            S.append(torch.nn.functional.pad(x,(0,0,to_pad,0),
                    mode="constant",value=0))

    ret = {}
    ret["x"] = torch.stack(Z)
    ret["sys_logs"] = torch.stack(S)
    ret["info"] = infos
    print(X)
    print(ret)
    pdb.set_trace()
    return ret

# PERCENTILES_TO_SAVE = [50, 99]
PERCENTILES_TO_SAVE = []
def percentile_help(q):
    def f(arr):
        return np.percentile(arr, q)
    return f

def qloss_torch(yhat, ytrue):
    min_est = np.array([0.1]*len(yhat))
    assert yhat.shape == ytrue.shape
    min_est = torch.tensor(min_est, dtype=torch.float)

    ytrue = torch.max(ytrue, min_est)
    yhat = torch.max(yhat, min_est)

    errors = torch.max( (ytrue / yhat), (yhat / ytrue))
    return torch.mean(errors)

class LinuxNN(LatencyPredictor):
    def __init__(self, *args, cfg={}, **kwargs):
        self.exp_version = None
        self.kwargs = kwargs
        for k, val in kwargs.items():
            self.__setattr__(k, val)

        self.cfg = cfg
        for k, val in cfg["common"].items():
            self.__setattr__(k, val)

        # self.collate_fn = collate_fn_gcn
        # self.collate_fn = collate_fn_gcn2
        self.collate_fn = None

        if self.loss_fn_name == "mse":
            self.loss_fn = torch.nn.MSELoss()
        elif self.loss_fn_name == "qerr":
            self.loss_fn = qloss_torch
        else:
            assert False

        self.eval_fns = []
        eval_fn_names = self.eval_fn_names.split(",")
        for l in eval_fn_names:
            self.eval_fns.append(get_eval_fn(l))

        self.summary_funcs = [np.mean]
        self.summary_types = ["mean"]

        for q in PERCENTILES_TO_SAVE:
            self.summary_funcs.append(percentile_help(q))
            self.summary_types.append("percentile:{}".format(str(q)))

        self.clip_gradient = None
        self.cur_stats = defaultdict(list)
        self.log_stat_fmt = "{samples_type}-{loss_type}"

    def _init_net(self):
        if self.arch == "factorized":
            self.net = FactorizedLinuxNet(self.cfg,
                    self.featurizer.num_features,
                    self.featurizer.num_global_features,
                    self.featurizer.num_syslog_features,
                    )
        else:
            assert False

        self.net = self.net.to(device)

    def periodic_eval(self, samples_type):

        start = time.time()
        dl = self.eval_loaders[samples_type]

        res,y = self._eval_loader(self.eval_ds[samples_type], dl)
        # print(y)
        # print(res)
        # pdb.set_trace()
        losses = []

        for eval_fn in self.eval_fns:
            errors = eval_fn.eval(res, y)

            self.log(errors, eval_fn.__str__(), samples_type)
            losses.append(np.round(np.mean(errors), 2))

        tot_time = round(time.time()-start, 2)
        stat_update = "{}:took {}, ".format(samples_type, tot_time)

        for i, eval_fn in enumerate(self.eval_fns):
            stat_update += eval_fn.__str__() + ": " + \
                bcolors.OKBLUE+str(losses[i])+bcolors.ENDC + "; "

        print(stat_update)

    def _train_one_epoch(self):
        start = time.time()
        epoch_losses = []

        for bidx, data in enumerate(self.traindl):
            y = torch.tensor(data["y"], dtype=torch.float32).to(device)
            yhat = self.net(data)

            if len(yhat.shape) > len(y.shape):
                yhat = yhat.squeeze()
            assert y.shape == yhat.shape
            loss = self.loss_fn(yhat, y)
            epoch_losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()

            if self.clip_gradient is not None:
                clip_grad_norm_(self.net.parameters(), self.clip_gradient)
            self.optimizer.step()

        if self.epoch % 2 == 0:
            print("epoch {}, loss: {}, epoch took: {}".format(\
                self.epoch, np.mean(epoch_losses), time.time()-start))

        self.log(epoch_losses, "train_loss", "train")

        if self.cfg["sys_net"]["save_weights"]:
            torch.save(self.net.sys_net.state_dict(),
                    self.cfg["sys_net"]["pretrained_fn"])
            print("saved sys net: ", self.cfg["sys_net"]["pretrained_fn"])

            if hasattr(self.net, "fact_net"):
                fname = self.cfg["sys_net"]["pretrained_fn"].replace("/",
                                                                     "/fact_")
                torch.save(self.net.fact_net.state_dict(),
                        fname)
                print("saved fact net: ", fname)


    def log(self, losses, loss_type, samples_type):
        for i, func in enumerate(self.summary_funcs):
            loss = func(losses)
            self.cur_stats["epoch"].append(self.epoch)
            self.cur_stats["loss_type"].append(loss_type)
            self.cur_stats["loss"].append(loss)
            self.cur_stats["summary_type"].append(self.summary_types[i])
            self.cur_stats["samples_type"].append(samples_type)
            stat_name = self.log_stat_fmt.format(
                    samples_type = samples_type,
                    loss_type = loss_type)

            # if self.summary_types[i] == "mean" and \
                    # self.use_wandb:
            if self.use_wandb:
                wandb.log({stat_name: loss, "epoch":self.epoch})

    def setup_workload(self, kind, df, sys_logs):
        if len(df) != 0:
            ds = LinuxJobDataset(df,
                    sys_logs,
                    self.featurizer,
                    self.cfg["sys_net"],
                    subplan_ests=self.subplan_ests,
                    )
            dl = torch.utils.data.DataLoader(ds,
                    batch_size=self.batch_size,
                    shuffle=False, collate_fn=self.collate_fn)

            self.eval_ds[kind] = ds
            self.eval_loaders[kind] = dl

    def train(self, df, sys_logs, featurizer,
            test=[],
            new_env_seen = [], new_env_unseen = [],
            ):

        self.featurizer = featurizer

        self.ds = LinuxJobDataset(df,
                sys_logs,
                self.featurizer,
                self.cfg["sys_net"],
                subplan_ests=self.subplan_ests,
                )

        self.traindl = torch.utils.data.DataLoader(self.ds,
                batch_size=self.batch_size,
                shuffle=True, collate_fn=self.collate_fn,
                )

        self.eval_loaders = {}
        self.eval_ds = {}

        self.eval_ds["train"] = self.ds
        self.eval_loaders["train"] = torch.utils.data.DataLoader(self.ds,
                batch_size=self.batch_size,
                shuffle=False, collate_fn=self.collate_fn)

        self.setup_workload("test", test, sys_logs)
        self.setup_workload("new_env_seen", new_env_seen, sys_logs)
        self.setup_workload("new_env_unseen", new_env_unseen, sys_logs)

        self._init_net()
        print(self.net)

        if self.cfg["sys_net"]["pretrained"]:
            print("Going to use pretrained system model from: ",
                    self.cfg["sys_net"]["pretrained_fn"])

            self.net.sys_net.load_state_dict(torch.load(
                    self.cfg["sys_net"]["pretrained_fn"]))

            for parameter in self.net.sys_net.parameters():
                parameter.requires_grad = False

            if hasattr(self.net, "fact_net") and \
                    self.cfg["factorized_net"]["pretrained"]:

                fname = self.cfg["sys_net"]["pretrained_fn"].replace("/",
                                                                     "/fact_")

                print("Going to use pretrained factorized head from: ",
                        fname)
                self.net.fact_net.load_state_dict(torch.load(
                    fname))

                for parameter in self.net.fact_net.parameters():
                    parameter.requires_grad = False

        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr,
                weight_decay=self.weight_decay)

        # self._save_embeddings(["train", "test"])
        # pdb.set_trace()

        for self.epoch in range(self.num_epochs):
            if self.epoch % self.eval_epoch == 0:
                for st in self.eval_loaders.keys():
                    self.periodic_eval(st)

            self._train_one_epoch()

    def _save_embeddings(self, sample_types):

        embeddings = []
        for st in sample_types:
            dl = self.eval_loaders[st]
            with torch.no_grad():
                for data in dl:
                    xsys = self.net.sys_net(data)
                    for bi,info in enumerate(data["info"]):
                        embeddings.append((info, xsys[bi].cpu().numpy()))

        # efn = self.cfg["sys_net"]["pretrained_fn"]
        # efn = efn.replace("models/", "embeddings/")
        # efn = efn.replace("models2/", "embeddings/")
        # efn = efn.replace(".wt", ".pkl")
        efn = "./embeddings/avg_stack.pkl"
        print("writing out embeddings to: ", efn)
        with open(efn, "wb") as f:
            pickle.dump(embeddings, f,
                            protocol=3)

    def _eval_loader(self, ds, dl):
        res = []
        trueys = []

        with torch.no_grad():
            for data in dl:
                yhat = self.net(data)
                y = torch.tensor(data["y"], dtype=torch.float32).to(device)

                if len(yhat.shape) > len(y.shape):
                    yhat = yhat.squeeze()

                # y = data.y
                # y = y.item()
                # trueys.append(self.featurizer.unnormalizeY(y))
                assert yhat.shape == y.shape
                for yh in yhat:
                    res.append(yh.item())
                for truey in y:
                    trueys.append(truey.item())

                # for gi in range(data["graph"].num_graphs):
                    # trueys.append(self.featurizer.unnormalizeY(y[gi].item()))

                # if self.subplan_ests:
                    # assert False
                # else:
                    # # yh = yhat.item()
                    # # res.append(self.featurizer.unnormalizeY(yh))
                    # for gi in range(data["graph"].num_graphs):
                        # res.append(self.featurizer.unnormalizeY(yhat[gi].item()))

        # self.net.train()
        # self.net.gcn_net.train()
        return res,trueys

    def test(self, df, sys_logs):
        ds = LinuxJobDataset(df,
                sys_logs,
                self.featurizer,
                self.cfg["sys_net"],
                subplan_ests=self.subplan_ests)
        dl = torch.utils.data.DataLoader(ds,
                batch_size=self.batch_size,
                shuffle=False, collate_fn=self.collate_fn)
        ret,_ = self._eval_loader(ds, dl)

        return ret
