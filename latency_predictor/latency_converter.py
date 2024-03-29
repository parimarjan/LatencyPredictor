import torch
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_
import torch.nn.functional as F
import torch_geometric
from torch.optim.lr_scheduler import *

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
from collections import defaultdict

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

# PERCENTILES_TO_SAVE = [50, 99]
PERCENTILES_TO_SAVE = []
def percentile_help(q):
    def f(arr):
        return np.percentile(arr, q)
    return f

def set_and_store_grad_state(model, exception_module_name="Z"):
    original_grad_states = {}
    for name, param in model.named_parameters():
        # Store the original requires_grad state
        original_grad_states[name] = param.requires_grad
        # If the current parameter is not the exception, set its requires_grad to False
        if not name.endswith(exception_module_name):
            param.requires_grad = False
        else:
            param.requires_grad = True
    return original_grad_states

def restore_grad_state(model, original_grad_states):
    for name, param in model.named_parameters():
        param.requires_grad = original_grad_states[name]

def collate_fn_gcn(Z):
    # return torch_geometric.data.Batch.from_data_list(Z).to(device)
    return torch_geometric.data.Batch.from_data_list(Z)

def collate_fn_gcn2(X):
    Z = [x["graph"] for x in X]
    # S = [x["sys_logs"] for x in X]
    # infos = [x["info"] for x in X]
    # Ys = [x["y"] for x in X]
    Ys = []
    infos = []
    S = []
    heuristic_feats = []
    hists = []

    ret = {}
    ret["sys_logs"] = torch.zeros(len(X), X[0]["sys_logs"].shape[0],
            X[0]["sys_logs"].shape[1])

    for ci,curx in enumerate(X):
        x = curx["sys_logs"]
        infos.append(curx["info"])
        Ys.append(curx["y"])

        if "heuristic_feats" in curx:
            heuristic_feats.append(curx["heuristic_feats"])
        if "history" in curx:
            hists.append(curx["history"])

        if len(x.shape) == 1:
            S.append(x)
        else:
            if x.shape[0] >= 95:
                # temporary
                rows = x.shape[1]
                to_pad = MAX_LOG_LEN-rows
                S.append(torch.nn.functional.pad(x,(0,0,0,to_pad),
                        mode="constant",value=0))
            else:
                rows = x.shape[0]
                to_pad = MAX_LOG_LEN-rows
                if to_pad < 0:
                    S.append(x)
                else:
                    S.append(torch.nn.functional.pad(x,(0,0,to_pad,0),
                            mode="constant",value=0))

    ret["graph"] = torch_geometric.data.Batch.from_data_list(Z).to(device)

    ## is this too slow?
    ret["sys_logs"] = torch.stack(S)

    # ret["sys_logs"] = torch.zeros(len(S), S[0].shape[0], S[0].shape[1])
    # for bi,b in enumerate(S):
        # ret["sys_logs"][bi] = b

    ret["info"] = infos
    ret["y"] = torch.tensor(Ys, dtype=torch.float)

    ret["heuristic_feats"] = torch.stack(heuristic_feats)
    ret["history"] = torch.stack(hists)

    return ret

def qloss_torch(yhat, ytrue):
    min_est = np.array([MIN_EST]*len(yhat))
    assert yhat.shape == ytrue.shape
    min_est = torch.tensor(min_est, dtype=torch.float,
            device="cuda:0")

    ytrue = torch.max(ytrue, min_est)
    yhat = torch.max(yhat, min_est)

    errors = torch.max( (ytrue / yhat), (yhat / ytrue))
    return torch.mean(errors)

def relloss_torch(yhat, ytrue):
    min_est = np.array([MIN_EST]*len(yhat))
    # min_est = torch.tensor(min_est, dtype=torch.float)
    min_est = torch.tensor(min_est, dtype=torch.float,
            device="cuda:0")
    yhat = torch.max(yhat, min_est)
    ytrue = torch.max(ytrue, min_est)
    relative_errors = torch.abs(yhat - ytrue) / (torch.abs(ytrue))

    return torch.mean(relative_errors)

class LatencyConverter(LatencyPredictor):
    def __init__(self, *args, cfg={}, **kwargs):
        self.exp_version = None
        self.kwargs = kwargs

        self.cfg = cfg
        for k, val in cfg["common"].items():
            self.__setattr__(k, val)

        for k, val in kwargs.items():
            self.__setattr__(k, val)

        # self.collate_fn = collate_fn_gcn
        # self.collate_fn = collate_fn_gcn2
        self.collate_fn = None

        if self.loss_fn_name == "mse":
            self.loss_fn = torch.nn.MSELoss()
        elif self.loss_fn_name == "ae":
            self.loss_fn = torch.nn.L1Loss()
        elif self.loss_fn_name == "qerr":
            self.loss_fn = qloss_torch
        elif self.loss_fn_name == "relloss":
            self.loss_fn = relloss_torch
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
        num_cols = self.featurizer.num_syslog_features + 5
        max_tokens = int(self.cfg["sys_net"]["log_prev_secs"] / 10)*2 + 1

        # num_cols = self.featurizer.num_syslog_features*2 + 5
        # max_tokens = int(self.cfg["sys_net"]["log_prev_secs"] / 10)

        self.net = TransformerLogs(
                num_cols,
                1,
                self.cfg["sys_net"]["num_layers"],
                self.cfg["sys_net"]["hl"],
                self.cfg["sys_net"]["num_heads"],
                max_tokens,
                self.cfg["sys_net"]["max_pool"],
                self.layernorm,
                self.cfg["sys_net"]["dropout"],
                data_field="x",
                )

        self.net = self.net.to(device)

    def periodic_eval(self, samples_type):

        start = time.time()
        dl = self.eval_loaders[samples_type]

        res,y,_ = self._eval_loader(self.eval_ds[samples_type], dl)
        losses = []

        lterrs = defaultdict(list)

        for eval_fn in self.eval_fns:
            errors = eval_fn.eval(res, y)

            self.log(errors, eval_fn.__str__(), samples_type)
            losses.append(np.round(np.mean(errors), 2))

            # if "Q" in str(eval_fn) and self.epoch % 10 == 0:
                # for i,err in enumerate(errors):
                    # lterrs[infos[i]["lt_type"]].append(err)
                # for lt,vals in lterrs.items():
                    # print("{}: {}".format(lt,np.round(np.mean(vals), 2)))

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
            y = data["y"].float().to(device)
            yhat = self.net(data)

            if len(yhat.shape) > len(y.shape):
                yhat = yhat.squeeze()
            assert y.shape == yhat.shape
            loss = self.loss_fn(yhat, y)

            self.optimizer.zero_grad()
            loss.backward()

            epoch_losses.append(loss.detach().item())

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

            # if self.summary_types[i] == "mean" and \
                    # self.use_wandb:
            if self.use_wandb:
                wandb.log({stat_name: loss, "epoch":self.epoch})

    def setup_workload(self, kind, plans, sys_logs, df):
        if len(plans) != 0:
            ds = LatencyConverterDataset(plans,
                    df,
                    sys_logs,
                    self.featurizer,
                    self.cfg["sys_net"],
                    subplan_ests=self.subplan_ests,
                    )

            dl = torch.utils.data.DataLoader(ds,
                    # batch_size= self.batch_size,
                    batch_size = 64,
                    shuffle=False,
                    collate_fn=self.collate_fn,
                    drop_last=True,
                    # num_workers=4,
                    )

            self.eval_ds[kind] = ds
            self.eval_loaders[kind] = dl

    def train(self, train_plans, sys_logs, featurizer,
            test=[],
            new_env_seen = [], new_env_unseen = [],
            train_df = None, test_df = None,
            unseen_df = None,
            ):

        self.featurizer = featurizer

        self.ds = LatencyConverterDataset(train_plans,
                train_df,
                sys_logs,
                self.featurizer,
                self.cfg["sys_net"],
                subplan_ests=self.subplan_ests,
                )

        self.traindl = torch.utils.data.DataLoader(self.ds,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=self.collate_fn,
                drop_last=True,
                )

        self.eval_loaders = {}
        self.eval_ds = {}

        self.eval_ds["train"] = self.ds
        self.eval_loaders["train"] = torch.utils.data.DataLoader(self.ds,
                batch_size=self.batch_size,
                shuffle=False, collate_fn=self.collate_fn,
                drop_last=True,
                )

        self.setup_workload("test", test, sys_logs, test_df)
        # self.setup_workload("new_env_seen", new_env_seen, sys_logs)

        self.setup_workload("new_env_unseen", new_env_unseen, sys_logs,
                            unseen_df)

        # if len(test) != 0:
            # ds = QueryPlanDataset(test,
                    # sys_logs,
                    # self.featurizer,
                    # self.cfg["sys_net"],
                    # subplan_ests=self.subplan_ests,
                    # )
            # dl = torch.utils.data.DataLoader(ds,
                    # batch_size=self.batch_size,
                    # shuffle=False, collate_fn=self.collate_fn)

            # self.eval_ds["test"] = ds
            # self.eval_loaders["test"] = dl

        ## TODO: initialize for other datasets

        self._init_net()
        print(self.net)

        # for name, param in self.net.named_parameters():
            # print(name, param.shape)

        if self.cfg["sys_net"]["pretrained"] and \
                hasattr(self.net, "sys_net"):
            print("Going to use pretrained system model from: ",
                    self.cfg["sys_net"]["pretrained_fn"])

            self.net.sys_net.load_state_dict(torch.load(
                    self.cfg["sys_net"]["pretrained_fn"],
					map_location="cpu",
                    ))

            for parameter in self.net.sys_net.parameters():
                parameter.requires_grad = False

            if hasattr(self.net, "fact_net") and \
                    self.cfg["factorized_net"]["pretrained"]:

                fname = self.cfg["sys_net"]["pretrained_fn"].replace("/",
                                                                     "/fact_")

                print("Going to use pretrained factorized head from: ",
                        fname)
                self.net.fact_net.load_state_dict(torch.load(
                    fname,
                    map_location="cpu",
                    ))

                for parameter in self.net.fact_net.parameters():
                    parameter.requires_grad = False

            if hasattr(self.net, "hist_net") and \
                    self.cfg["hist_net"]["pretrained"]:
                fname = self.cfg["sys_net"]["pretrained_fn"].replace("/",
                                                                     "/hist_")

                print("Going to use pretrained hist net from: ",
                        fname)
                self.net.hist_net.load_state_dict(torch.load(
                    fname,
                    map_location="cpu",
                    ))

                if self.cfg["hist_net"]["pretrained"] == 1:
                    for parameter in self.net.hist_net.parameters():
                        parameter.requires_grad = False

            if self.cfg["plan_net"]["pretrained"]:
                fname = self.cfg["sys_net"]["pretrained_fn"].replace("/",
                                                                     "/gcn_")

                print("Going to use pretrained gcn net from: ",
                        fname)
                self.net.gcn_net.load_state_dict(torch.load(
                    fname,
                    map_location="cpu",
                    ))

                if self.cfg["plan_net"]["pretrained"] == 2:
                    for parameter in self.net.gcn_net.parameters():
                        parameter.requires_grad = False


            if self.cfg["latent_variable"] and \
                    self.cfg["latent_inference"]:
                fname = self.cfg["sys_net"]["pretrained_fn"].replace("/",
                                                                     "/latent_")
                print("Going to load latent Z from: ",
                        fname)
                self.net.Z = torch.load(
                    fname,
                    map_location="cpu",
                    )
                self.net.Z.requires_grad = False
                self.net.Z = nn.Parameter(self.net.Z.to(device))

        for name, param in self.net.named_parameters():
            if param.requires_grad:
                print(name)

        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr,
                weight_decay=self.weight_decay)

        if self.lrscheduler:
            self.optimizer = torch.optim.AdamW(self.net.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay)
            train_scheduler = CosineAnnealingLR(self.optimizer, self.num_epochs)
            number_warmup_epochs = self.lrscheduler

            def warmup(current_step):
                print(current_step / float(number_warmup_epochs))
                return current_step / float(number_warmup_epochs)

            warmup_scheduler = LambdaLR(self.optimizer, lr_lambda=warmup)
            self.scheduler = SequentialLR(self.optimizer,
                    [warmup_scheduler, train_scheduler], [number_warmup_epochs])

        # self._save_embeddings(["train"])
        # self._save_embeddings(["train", "test"])
        # pdb.set_trace()

        if self.cfg["sys_net"]["arch"] == "transformer":
            exp_name = self.get_exp_name()
            rdir = os.path.join(self.result_dir,
                    exp_name)
            if os.path.exists(rdir):
                rfn = os.path.join(rdir, "env_net_normalizers.pkl")
                with open(rfn, 'wb') as handle:
                    pickle.dump(self.featurizer.sys_norms, handle)
                print("saved sys net normalizers: ", rfn)

        for self.epoch in range(self.num_epochs):
            if self.epoch % self.eval_epoch == 0:
                for st in self.eval_loaders.keys():
                    # if st == "train":
                        # continue
                    self.periodic_eval(st)

            self._train_one_epoch()

            if self.lrscheduler:
                self.scheduler.step()
                print(f"Epoch: {self.epoch+1}, Learning rate: {self.scheduler.get_last_lr()[0]}")

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
        # efn = "./embeddings/model_bg.pkl"
        efn = "./embeddings/avg_bg_m4.pkl"
        print("writing out embeddings to: ", efn)
        with open(efn, "wb") as f:
            pickle.dump(embeddings, f,
                            protocol=3)

    def _eval_loader(self, ds, dl):
        res = []
        trueys = []
        infos = []

        for di,data in enumerate(dl):
            yhat = self.net(data)
            y = data["y"]
            if len(yhat.shape) > len(y.shape):
                yhat = yhat.squeeze()

            assert yhat.shape == y.shape

            for gi in range(len(y)):
                trueys.append(self.featurizer.unnormalizeY(y[gi].item()))

            for gi in range(len(yhat)):
                res.append(self.featurizer.unnormalizeY(yhat[gi].item()))

        # Printing the formatted distribution in one line
        print(f"Min: {np.min(res):.4f}, 25th Percentile: {np.percentile(res, 25):.4f}, Median: {np.median(res):.4f}, 75th Percentile: {np.percentile(res, 75):.4f}, Max: {np.max(res):.4f}")

        return res,trueys,infos

    def test(self, plans, sys_logs):
        ds = QueryPlanDataset(plans,
                sys_logs,
                self.featurizer,
                self.cfg["sys_net"],
                subplan_ests=self.subplan_ests)
        dl = torch.utils.data.DataLoader(ds,
                batch_size=self.batch_size,
                drop_last=True,
                shuffle=False, collate_fn=self.collate_fn)
        ret,_,_ = self._eval_loader(ds, dl)

        return ret
