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

INFERENCE_SAMPLES=20
FINETUNE_EVERY_BATCH=True

SKIP_FIRST=False
SKIP_OVERLAP=False

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
            # assert False
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

class NN(LatencyPredictor):
    def __init__(self, *args, cfg={}, **kwargs):
        self.exp_version = None
        self.kwargs = kwargs

        self.cfg = cfg
        for k, val in cfg["common"].items():
            self.__setattr__(k, val)

        for k, val in kwargs.items():
            self.__setattr__(k, val)
        self.fact_arch = self.cfg["factorized_net"]["arch"]

        # self.collate_fn = collate_fn_gcn
        self.collate_fn = collate_fn_gcn2

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
        if self.arch in ["gcn","gat"]:
            self.net = SimpleGCN(self.featurizer.num_features,
                    self.featurizer.num_global_features,
                    self.hl1, self.cfg["plan_net"]["num_conv_layers"],
                    final_act=self.final_act,
                    subplan_ests = self.subplan_ests,
                    out_feats=1,
                    dropout=self.cfg["plan_net"]["dropout"],
                    arch=self.cfg["plan_net"]["arch"],
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
                    1, 4,
                    self.cfg["sys_net"]["hl"],
                    self.cfg["sys_net"]["num_heads"],
                    int(self.cfg["sys_net"]["log_prev_secs"] / 10),
                    self.cfg["sys_net"]["max_pool"],
                    self.layernorm,
                    self.cfg["hist_net"]["dropout"],
                    )
        elif self.arch == "transformerhistory":
            self.net = TransformerLogs(
                    5, 1,
                    self.cfg["hist_net"]["num_layers"],
                    self.cfg["hist_net"]["hl"],
                    self.cfg["hist_net"]["num_heads"],
                    NUM_HIST,
                    self.cfg["hist_net"]["max_pool"],
                    self.layernorm,
                    self.cfg["hist_net"]["dropout"],
                    data_field="history",
                    )

        elif self.arch == "factorized":
            if self.featurizer.sys_seq_kind == "rows":
                self.net = FactorizedLatencyNet(self.cfg,
                        self.featurizer.num_features,
                        self.featurizer.num_global_features,
                        self.featurizer.num_syslog_features,
                        MAX_LOG_LEN,
                        self.layernorm,
                        NUM_HIST,
                        )
            elif "col" in self.featurizer.sys_seq_kind:
                self.net = FactorizedLatencyNet(self.cfg,
                        self.featurizer.num_features,
                        self.featurizer.num_global_features,
                        MAX_LOG_LEN + self.featurizer.num_syslog_features,
                        self.featurizer.num_syslog_features,
                        self.layernorm,
                        NUM_HIST,
                        )
        else:
            assert False

        self.net = self.net.to(device)

    def plot_distributions(self, samples_type):
        dl = self.eval_loaders[samples_type]

    def periodic_eval(self, samples_type):

        start = time.time()
        dl = self.eval_loaders[samples_type]

        res,y,infos = self._eval_loader(self.eval_ds[samples_type], dl)
        assert len(res) == len(y)
        losses = []

        lterrs = defaultdict(list)

        for eval_fn in self.eval_fns:
            errors = eval_fn.eval(res, y)

            self.log(errors, eval_fn.__str__(), samples_type)
            losses.append(np.round(np.mean(errors), 2))

            if "Q" in str(eval_fn) and self.epoch % 10 == 0:
                for i,err in enumerate(errors):
                    lterrs[infos[i]["lt_type"]].append(err)
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
            y = data["y"].to(device)
            yhat = self.net(data)

            if self.subplan_ests:
                assert False

            if self.fact_arch == "flow":
                self.optimizer.zero_grad()
                context = yhat

                # Negative log likelihood
                ln_p_y_given_x = self.net.dist_y_given_x.\
                        condition(context).log_prob(y.unsqueeze(dim=1))
                nll_loss = -ln_p_y_given_x.mean()

                # Update loss and gradients
                nll_loss.backward()
                self.optimizer.step()
                self.net.dist_y_given_x.clear_cache()
                epoch_losses.append(nll_loss.detach().item())
            else:
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
            if self.fact_arch == "flow":
                print("epoch {}, NLL loss: {}, epoch took: {}".format(\
                    self.epoch, np.mean(epoch_losses), time.time()-start))
            else:
                print("epoch {}, loss: {}, epoch took: {}".format(\
                    self.epoch, np.mean(epoch_losses), time.time()-start))

        self.log(epoch_losses, "train_loss", "train")

        if hasattr(self.net, "sys_net") and self.net.sys_net != None:
            exp_name = self.get_exp_name()
            rdir = os.path.join(self.result_dir, exp_name)
            # if os.path.exists(rdir):
                # rfn = os.path.join(rdir, "env_net.wt")
                # torch.save(self.net.sys_net.state_dict(),
                        # rfn)
                # print("saved sys net: ", rfn)

        if self.cfg["sys_net"]["save_weights"] and \
                self.cfg["sys_net"]["arch"] == "transformer" and \
                hasattr(self.net, "sys_net"):
            torch.save(self.net.sys_net.state_dict(),
                    self.cfg["sys_net"]["pretrained_fn"])
            print("saved sys net: ", self.cfg["sys_net"]["pretrained_fn"])

            if hasattr(self.net, "fact_net") or \
                    hasattr(self.net, "transforms"):
                fname = self.cfg["sys_net"]["pretrained_fn"].replace("/",
                                                                     "/fact_")
                if self.fact_arch == "flow":
                    transform_parameters = [t.state_dict() for t \
                            in self.net.transforms]
                    torch.save(transform_parameters, fname)
                else:
                    torch.save(self.net.fact_net.state_dict(),
                            fname)

            if hasattr(self.net, "gcn_net"):
                fname = self.cfg["sys_net"]["pretrained_fn"].replace("/",
                                                                     "/gcn_")
                torch.save(self.net.gcn_net.state_dict(),
                        fname)
                # print("saved gcn net: ", fname)

            if hasattr(self.net, "Z"):
                fname = self.cfg["sys_net"]["pretrained_fn"].replace("/",
                                                                     "/latent_")
                torch.save(self.net.Z,
                        fname)
                # print("saved latent Z: ", fname)

            if hasattr(self.net, "hist_net"):
                fname = self.cfg["sys_net"]["pretrained_fn"].replace("/",
                                                                     "/hist_")
                torch.save(self.net.hist_net.state_dict(),
                        fname)
                print("saved hist net: ", fname)

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

    def setup_workload(self, kind, plans, sys_logs):
        if len(plans) != 0:
            ds = QueryPlanDataset(plans,
                    sys_logs,
                    self.featurizer,
                    self.cfg["sys_net"],
                    subplan_ests=self.subplan_ests,
                    )

            dl = torch.utils.data.DataLoader(ds,
                    batch_size= self.batch_size,
                    # batch_size = 8,
                    shuffle=False,
                    collate_fn=self.collate_fn,
                    drop_last=False,
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

        # if self.fact_arch == "flow":
            # assert self.featurizer.y_normalizer == "std"
        self.ds = QueryPlanDataset(train_plans,
                sys_logs,
                self.featurizer,
                self.cfg["sys_net"],
                subplan_ests=self.subplan_ests,
                )

        self.traindl = torch.utils.data.DataLoader(self.ds,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=self.collate_fn,
                drop_last=False,
                # num_workers = 4,
                )

        self.eval_loaders = {}
        self.eval_ds = {}

        self.eval_ds["train"] = self.ds
        self.eval_loaders["train"] = torch.utils.data.DataLoader(self.ds,
                batch_size=self.batch_size,
                shuffle=False, collate_fn=self.collate_fn,
                drop_last=False,
                )

        self.setup_workload("test", test, sys_logs)
        self.setup_workload("new_env_seen", new_env_seen, sys_logs)
        self.setup_workload("new_env_unseen", new_env_unseen, sys_logs)

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
        # pdb.set_trace()

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

            if (hasattr(self.net, "fact_net") or \
                    hasattr(self.net, "transforms")) and \
                    self.cfg["factorized_net"]["pretrained"]:

                fname = self.cfg["sys_net"]["pretrained_fn"].replace("/",
                                                                     "/fact_")
                print("Going to use pretrained factorized head from: ",
                        fname)

                if self.fact_arch == "flow":
                    base_dist = dist.Normal(torch.zeros(1, device=device),
                            torch.ones(1, device=device))
                    loaded_transform_parameters = torch.load(fname,
                            map_location="cpu")
                    assert len(loaded_transform_parameters)==len(self.net.transforms)
                    for t, state_dict in zip(self.net.transforms,
                            loaded_transform_parameters):
                        t.load_state_dict(state_dict)
                        for parameter in t.parameters():
                            parameter.requires_grad = False
                    self.net.dist_y_given_x = dist.\
                            ConditionalTransformedDistribution(base_dist,
                                    self.net.transforms)
                else:
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

        if self.fact_arch == "flow":
            self.optimizer = torch.optim.AdamW(
                list(self.net.parameters()) +
                [p for t in self.net.transforms for p in t.parameters()],  # Include parameters of all transforms
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

        else:
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
        # self._save_embeddings(["test"])
        # pdb.set_trace()

        # self._plot_distributions("train")

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
            if self.epoch % self.eval_epoch == 0 \
                    and self.epoch != 0:
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

        # efn = "./embeddings/stack_single.pkl"
        # efn = "./embeddings/stack_single_avg.pkl"
        efn = "./embeddings/imdb_single_avg.pkl"
        # efn = "./embeddings/imdb_single.pkl"
        # efn = "./embeddings/imdb_all_avg.pkl"
        # efn = "./embeddings/imdb_all.pkl"
        print("writing out embeddings to: ", efn)
        with open(efn, "wb") as f:
            pickle.dump(embeddings, f,
                            protocol=3)

    def _eval_loader(self, ds, dl):
        if self.cfg["latent_inference"]:
            return self._eval_loader_latent(ds, dl)

        # if "finetune_inference" in self.cfg["factorized_net"] and \
                # self.cfg["factorized_net"]["finetune_inference"]:
            # return self._eval_loader_flow_finetune(ds, dl)

        if "finetune_inference" in self.cfg["factorized_net"] and \
                self.cfg["factorized_net"]["finetune_inference"]:
            return self._eval_loader_flow_latent(ds, dl)

        prev_instance = None
        latent = None

        res = []
        trueys = []
        infos = []

        for di,data in enumerate(dl):
            if len(data["info"]) == 1:
                continue
            cur_instance = data["info"][0]["instance"]
            lt_type = data["info"][0]["lt_type"]
            if cur_instance != prev_instance and \
                    prev_instance is not None:
                latent = None

            iset = set()
            for curinfo in data["info"]:
                iset.add(curinfo["instance"])
            if len(iset) > 1 and SKIP_OVERLAP:
                print("Skipping batch because has multiple instances: ",
                        iset)
                continue

            prev_instance = cur_instance
            if latent is None and SKIP_FIRST:
                latent = 0.0
                print("skipping first batch of instance")
                continue

            yhat = self.net(data)
            y = data["y"]

            if self.fact_arch != "flow":
                if len(yhat.shape) > len(y.shape):
                    yhat = yhat.squeeze()
                assert yhat.shape == y.shape

            for gi in range(data["graph"].num_graphs):
                trueys.append(self.featurizer.unnormalizeY(y[gi].item()))
                infos.append(data["info"][gi])

            if self.subplan_ests:
                assert False
            for gi in range(data["graph"].num_graphs):
                if self.fact_arch == "flow":
                    single_context = yhat[gi].unsqueeze(0)
                    y_samples = self.net.dist_y_given_x.\
                            condition(single_context).sample((INFERENCE_SAMPLES,))
                    est = y_samples.mean().item()
                    est = self.featurizer.unnormalizeY(est)
                else:
                    est = self.featurizer.unnormalizeY(yhat[gi].item())

                if est < MIN_EST:
                    est = MIN_EST
                if est > MAX_EST:
                    est = MAX_EST

                res.append(est)

        print(f"True distribution, Min: {np.min(trueys):.4f}, 25th Percentile: {np.percentile(trueys, 25):.4f}, Median: {np.median(trueys):.4f}, 75th Percentile: {np.percentile(trueys, 75):.4f}, Max: {np.max(trueys):.4f}")


        print(f"Est distribution, Min: {np.min(res):.4f}, 25th Percentile: {np.percentile(res, 25):.4f}, Median: {np.median(res):.4f}, 75th Percentile: {np.percentile(res, 75):.4f}, Max: {np.max(res):.4f}")

        return res,trueys,infos

    def _load_flow_net(self, fname, req_grad=False):
        loaded_transform_parameters = torch.load(fname,
                map_location="cpu")
        for t, state_dict in zip(self.net.transforms,
                loaded_transform_parameters):
            t.load_state_dict(state_dict)
            for parameter in t.parameters():
                parameter.requires_grad = req_grad

        base_dist = dist.Normal(torch.zeros(1, device=device),
                torch.ones(1, device=device))
        self.net.dist_y_given_x = dist.\
                ConditionalTransformedDistribution(base_dist,
                        self.net.transforms)

    def _eval_loader_flow_latent(self, ds, dl):
        res = []
        trueys = []
        infos = []
        prev_instance = None
        prev_lt_type = None
        assert self.fact_arch == "flow"

        latent = None
        for di,data in enumerate(dl):
            cur_instance = data["info"][0]["instance"]
            lt_type = data["info"][0]["lt_type"]
            if cur_instance != prev_instance and \
                    prev_instance is not None:
                latent = None

            # FIXME: temporary
            iset = set()
            for curinfo in data["info"]:
                iset.add(curinfo["instance"])
            if len(iset) > 1 and SKIP_OVERLAP:
                print("Skipping batch because has multiple instances: ",
                        iset)
                continue

            prev_instance = cur_instance
            y = data["y"]
            context = self.net(data)

            # FIXME: just avoiding predictions on batch 1
            if latent is None and SKIP_FIRST:
                cur_obs = y.to(device).unsqueeze(1)
                assert len(self.net.transforms) == 1
                latent = self.net.transforms[0].condition(context).inv(cur_obs)
                print("skipping first batch of instance")
                continue

            for gi in range(data["graph"].num_graphs):
                trueys.append(self.featurizer.unnormalizeY(y[gi].item()))
                infos.append(data["info"][gi])

            if len(context.shape) > len(y.shape):
                context = context.squeeze()

            if latent is None:
                for gi in range(data["graph"].num_graphs):
                    single_context = context[gi].unsqueeze(0)
                    y_samples = self.net.dist_y_given_x.\
                            condition(single_context).sample((INFERENCE_SAMPLES,))
                    est = self.featurizer.unnormalizeY(y_samples.mean().item())
                    if est < MIN_EST:
                        est = MIN_EST
                    if est > MAX_EST:
                        est = MAX_EST
                    res.append(est)
            else:
                ## TODO: does taking mean even make sense?
                latent_mean = latent.mean()
                latent_mean_filled = torch.full(latent.shape, latent_mean.item())
                # print("Min: {}, Max: {}, Mean: {}".format(torch.min(latent),
                    # torch.max(latent), latent_mean))
                latent_mean_filled = latent_mean_filled.to(device)
                yhat = self.net.transforms[0].condition(context)(latent_mean_filled)
                for gi in range(data["graph"].num_graphs):
                    est = self.featurizer.unnormalizeY(yhat[gi].item())
                    if est < MIN_EST:
                        est = MIN_EST
                    if est > MAX_EST:
                        est = MAX_EST

                    res.append(est)

            # we always want to do this so the latent is being updated on the
            # previous batch;
            # assuming only one layer of transforms
            cur_obs = y.to(device).unsqueeze(1)
            assert len(self.net.transforms) == 1
            latent = self.net.transforms[0].condition(context).inv(cur_obs)

        print(f"True distribution, Min: {np.min(trueys):.4f}, 25th Percentile: {np.percentile(trueys, 25):.4f}, Median: {np.median(trueys):.4f}, 75th Percentile: {np.percentile(trueys, 75):.4f}, Max: {np.max(trueys):.4f}")


        print(f"Min: {np.min(res):.4f}, 25th Percentile: {np.percentile(res, 25):.4f}, Median: {np.median(res):.4f}, 75th Percentile: {np.percentile(res, 75):.4f}, Max: {np.max(res):.4f}")

        return res,trueys,infos

    def _eval_loader_flow_finetune(self, ds, dl):
        res = []
        trueys = []
        infos = []
        prev_instance = None
        assert self.fact_arch == "flow"

        # fname = self.cfg["sys_net"]["pretrained_fn"].replace("/",
                                                             # "/fact_")
        # self._load_flow_net(fname, req_grad=True)

        cur_net = copy.deepcopy(self.net)

        cur_trained = False
        for di,data in enumerate(dl):
            cur_instance = data["info"][0]["instance"]
            if cur_instance != prev_instance and \
                    prev_instance is not None:
                # print("going to train for new instance: ", cur_instance)
                cur_net = copy.deepcopy(self.net)
                cur_trained = False

            prev_instance = cur_instance
            y = data["y"]
            for gi in range(data["graph"].num_graphs):
                trueys.append(self.featurizer.unnormalizeY(y[gi].item()))
                infos.append(data["info"][gi])

            yhat = cur_net(data)
            if len(yhat.shape) > len(y.shape):
                yhat = yhat.squeeze()

            ## normalizing flow sampling + prediction
            for gi in range(data["graph"].num_graphs):
                single_context = yhat[gi].unsqueeze(0)
                y_samples = cur_net.dist_y_given_x.\
                        condition(single_context).sample((INFERENCE_SAMPLES,))
                est = self.featurizer.unnormalizeY(y_samples.mean().item())
                if est < MIN_EST:
                    est = MIN_EST
                if est > MAX_EST:
                    est = MAX_EST
                res.append(est)

            if not cur_trained or FINETUNE_EVERY_BATCH:
                ## training on the current batch
                z_optimizer = torch.optim.Adam(
                    [p for t in cur_net.transforms for p in t.parameters()],
                    lr=0.001,
                )

                # Training loop
                epochs = 75
                y = y.to(device)
                for epoch in range(epochs):
                    z_optimizer.zero_grad()
                    context = cur_net(data)
                    # Negative log likelihood
                    ln_p_y_given_x = cur_net.dist_y_given_x.\
                            condition(context).log_prob(y.unsqueeze(dim=1))
                    nll_loss = -ln_p_y_given_x.mean()

                    # pdb.set_trace()
                    # Update loss and gradients
                    nll_loss.backward()
                    z_optimizer.step()
                    cur_net.dist_y_given_x.clear_cache()

                    # if epoch % 25 == 0:
                        # print(f"Epoch [{epoch+1}/{epochs}], Loss: {nll_loss.item():.2f}")
                cur_trained = True

        # restore_grad_state(self.net, original_grad_states)
        # load the original fact net back
        # self._load_flow_net(fname, req_grad=False)

        # Printing the formatted distribution in one line
        print(f"Min: {np.min(res):.4f}, 25th Percentile: {np.percentile(res, 25):.4f}, Median: {np.median(res):.4f}, 75th Percentile: {np.percentile(res, 75):.4f}, Max: {np.max(res):.4f}")

        return res,trueys,infos

    def _eval_loader_latent(self, ds, dl):
        res = []
        trueys = []
        infos = []

        if self.cfg["latent_inference"] and \
                    hasattr(self.net, "Z") and \
                    self.cfg["factorized_net"]["pretrained"]:
            print("going to do latent inference on preceding batches")
            fname = self.cfg["sys_net"]["pretrained_fn"].replace("/",
                                                                 "/latent_")
            print("Going to load latent Z from: ",
                    fname)

            self.net.Z = torch.load(
                fname,
                map_location="cpu",
                )
            self.net.Z = nn.Parameter(self.net.Z.to(device))
            self.net.Z.requires_grad = True

        # TODO: make sure the rest of the model has requires grad turned off
        # to prevent memory usage
        original_grad_states = set_and_store_grad_state(self.net,
                exception_module_name="Z")

        prev_instance = None

        for di,data in enumerate(dl):

            cur_instance = data["info"][0]["instance"]
            if cur_instance != prev_instance and \
                    prev_instance is not None and \
                    hasattr(self.net, "z") and \
                    self.cfg["factorized_net"]["pretrained"]:
                self.net.Z = torch.load(
                    fname,
                    map_location="cpu",
                    )
                self.net.Z = nn.Parameter(self.net.Z.to(device))
                self.net.Z.requires_grad = True

            prev_instance = cur_instance

            yhat = self.net(data)
            y = data["y"]

            if len(yhat.shape) > len(y.shape):
                yhat = yhat.squeeze()

            assert yhat.shape == y.shape

            for gi in range(data["graph"].num_graphs):
                trueys.append(self.featurizer.unnormalizeY(y[gi].item()))
                infos.append(data["info"][gi])

            if self.subplan_ests:
                assert False
            else:
                # yh = yhat.item()
                # res.append(self.featurizer.unnormalizeY(yh))
                for gi in range(data["graph"].num_graphs):
                    est = self.featurizer.unnormalizeY(yhat[gi].item())
                    if est < MIN_EST:
                        est = MIN_EST
                    if est > MAX_EST:
                        est = MAX_EST
                    res.append(est)

            if self.cfg["latent_inference"] \
                    and hasattr(self.net, "Z"):
                # res, trueys are the preds, trainings; we want to train
                # ONLY the self.Z part, from scratch, just on the current
                # batch so we can use it on the next batch.

                # Create a separate optimizer for self.Z
                # Faster learning rate for Z
                # z_optimizer = torch.optim.SGD([self.net.Z], lr=0.001)
                # z_optimizer = torch.optim.Adam([self.net.Z], lr=0.001)
                # z_optimizer = torch.optim.SGD([self.net.Z], lr=0.001)
                z_optimizer = torch.optim.Adam([self.net.Z], lr=0.0001)
                # criterion = nn.MSELoss()
                criterion = qloss_torch

                # Training loop
                epochs = 100  # Number of times to update self.Z based on this batch
                y = y.to(device)
                for epoch in range(epochs):
                    # Zero the parameter gradients
                    z_optimizer.zero_grad()

                    # Forward pass
                    outputs = self.net(data)
                    if len(outputs.shape) > len(y.shape):
                        outputs = outputs.squeeze()

                    loss = criterion(outputs, y)

                    # Backward pass and optimization
                    loss.backward()
                    z_optimizer.step()

                    # if epoch % 25 == 0:
                        # print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

        restore_grad_state(self.net, original_grad_states)

        ## load the original Z back
        if self.arch == "factorized" and \
                    hasattr(self.net, "z") and \
                    self.cfg["factorized_net"]["pretrained"]:
            self.net.Z = torch.load(
                fname,
                map_location="cpu",
                )
            self.net.Z = nn.Parameter(self.net.Z.to(device))
            self.net.Z.requires_grad = False

        # print("min estimate: ", np.min(res))
        # print("max estimate: ", np.max(res))

        # Printing the formatted distribution in one line
        print(f"Min: {np.min(res):.4f}, 25th Percentile: {np.percentile(res, 25):.4f}, Median: {np.median(res):.4f}, 75th Percentile: {np.percentile(res, 75):.4f}, Max: {np.max(res):.4f}")

        return res,trueys,infos

    def test(self, plans, sys_logs, samples_type=None):
        ds = QueryPlanDataset(plans,
                sys_logs,
                self.featurizer,
                self.cfg["sys_net"],
                subplan_ests=self.subplan_ests)
        dl = torch.utils.data.DataLoader(ds,
                batch_size=2,
                drop_last=True,
                shuffle=False, collate_fn=self.collate_fn)
        ret,_,_ = self._eval_loader(ds, dl)

        return ret
