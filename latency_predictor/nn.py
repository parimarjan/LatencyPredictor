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

from latency_predictor.dataset import *
from latency_predictor.eval_fns import *
from latency_predictor.utils import *
from latency_predictor.nets import *
from latency_predictor.algs import *

import pickle
import wandb

PERCENTILES_TO_SAVE = [50, 99]
def percentile_help(q):
    def f(arr):
        return np.percentile(arr, q)
    return f

def collate_fn_gcn(Z):
    return torch_geometric.data.Batch.from_data_list(Z).to(device)

def qloss_torch(yhat, ytrue):
    assert yhat.shape == ytrue.shape
    errors = torch.max( (ytrue / yhat), (yhat / ytrue))
    return torch.mean(errors)

class NN(LatencyPredictor):
    def __init__(self, *args, cfg={}, **kwargs):
        self.exp_version = None
        self.kwargs = kwargs
        for k, val in kwargs.items():
            self.__setattr__(k, val)

        self.cfg = cfg
        for k, val in cfg["common"].items():
            self.__setattr__(k, val)

        self.collate_fn = collate_fn_gcn

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
        if self.arch == "gcn":
            self.net = SimpleGCN(self.featurizer.num_features,
                    self.featurizer.num_global_features,
                    self.hl1, self.num_conv_layers,
                    final_act=self.final_act,
                    subplan_ests = self.subplan_ests,
                    out_feats=1,
                    )
        elif self.arch == "avg":
            self.net = LogAvgRegression(
                    self.featurizer.num_syslog_features,
                    1, 4, self.hl1)
        elif self.arch == "tst":
            self.net = TSTLogs(
                    self.featurizer.num_syslog_features,
                    1, 4, self.hl1)
        elif self.arch == "transformerlogs":
            self.net = TransformerLogs(
                    self.featurizer.num_syslog_features,
                    1, 4, self.hl1)
        elif self.arch == "factorized":
            self.net = FactorizedLatencyNet(self.cfg,
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
        losses = []

        for eval_fn in self.eval_fns:
            errors = eval_fn.eval(res, y)

            self.log(errors, eval_fn.__str__(), samples_type)
            losses.append(np.mean(errors))

            # if hasattr(self, "update_best") and self.update_best:
                # # wandb.run.summary
                # loss_key = "Best-{}-{}-{}".format(str(eval_fn),
                                                   # samples_type,
                                                   # "mean")
                # wandb.run.summary[loss_key] = np.mean(errors)

        stat_update = "{}:".format(samples_type)
        for i, eval_fn in enumerate(self.eval_fns):
            stat_update += eval_fn.__str__() + ": " + str(losses[i]) + "; "
        print(stat_update)

    def _train_one_epoch(self):
        start = time.time()
        epoch_losses = []

        for bidx, data in enumerate(self.traindl):
            y = data.y.to(device)
            yhat = self.net(data)

            if self.subplan_ests:
                assert False
                # yhat = yhat.squeeze()
                # # yhat = yhat*(y != 0.0)
                # assert y.shape == yhat.shape
                # yhat = yhat[y != 0.0]
                # y = y[y != 0.0]
                # loss = self.loss_fn(yhat, y)
                # if self.subplan_sum_loss:
                    # ## not clear if this will add anything to it
                    # ysum = torch.sum(yhat)
                    # ytruesum = torch.sum(y)
                    # loss2 = self.loss_fn(ysum, ytruesum)
                    # loss = loss + loss2
            else:
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

            if self.summary_types[i] == "mean" and \
                    self.use_wandb:
                wandb.log({stat_name: loss, "epoch":self.epoch})

    def train(self, train_plans, sys_logs, featurizer,
            sys_log_feats,
            same_env_unseen=None,
            new_env_seen = None, new_env_unseen = None,
            ):

        self.featurizer = featurizer
        self.sys_log_feats = sys_log_feats

        self.ds = QueryPlanDataset(train_plans,
                sys_logs,
                self.featurizer,
                self.cfg["sys_net"],
                # self.sys_log_feats,
                subplan_ests=self.subplan_ests,
                )
        self.traindl = torch.utils.data.DataLoader(self.ds,
                batch_size=self.batch_size,
                shuffle=True, collate_fn=self.collate_fn)

        self.eval_loaders = {}
        self.eval_ds = {}

        self.eval_ds["train"] = self.ds
        self.eval_loaders["train"] = torch.utils.data.DataLoader(self.ds,
                batch_size=self.batch_size,
                shuffle=False, collate_fn=self.collate_fn)

        ## TODO: initialize for other datasets

        self._init_net()
        print(self.net)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr,
                weight_decay=self.weight_decay)

        for self.epoch in range(self.num_epochs):
            if self.epoch % self.eval_epoch == 0:
                self.periodic_eval("train")

            self._train_one_epoch()

    def _eval_loader(self, ds, dl):
        res = []
        trueys = []
        self.net.eval()

        with torch.no_grad():
            for data in dl:
                yhat = self.net(data)
                # yhat = yhat.squeeze()
                y = data.y
                # print(y.shape, yhat.shape)
                # pdb.set_trace()
                assert yhat.shape == y.shape
                y = y.item()
                trueys.append(self.featurizer.unnormalizeY(y))

                if self.subplan_ests:
                    assert False
                else:
                    yh = yhat.item()
                    res.append(self.featurizer.unnormalizeY(yh))

        self.net.train()
        return res,trueys

    def test(self, plans, sys_logs):
        ds = QueryPlanDataset(plans,
                sys_logs,
                self.featurizer,
                self.cfg["sys_net"],
                subplan_ests=self.subplan_ests)
        dl = torch.utils.data.DataLoader(ds,
                batch_size=self.batch_size,
                shuffle=False, collate_fn=self.collate_fn)
        ret,_ = self._eval_loader(ds, dl)

        return ret
