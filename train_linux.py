import argparse
import random
import time
import ntpath
import os
import pdb
from collections import defaultdict

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from latency_predictor.utils import *
from latency_predictor.algs import *
from latency_predictor.linux_nn import LinuxNN
from latency_predictor.eval_fns import *
# from latency_predictor.featurizer import *
from latency_predictor.linux_featurizer import *
import wandb
import logging
import csv
import yaml
import multiprocessing as mp

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)

def split_workload(df, cfg):

    split_kind = cfg.get("split_kind", "instance")
    # inum = cfg.get("instance_num", 1)
    inum = cfg["num_instances"]

    if split_kind == "instances":
        print("Random seed: ", cfg["seed"], " Num Instances: ", inum)
        random.seed(cfg["seed"])
        instances = list(set(df["lt_type"]))
        instances.sort()
        train_qinstances = random.sample(instances, inum)
        test_qinstances = [q for q in instances if q not in
                train_qinstances]
        train_df = df[df["lt_type"].isin(train_qinstances)]
        test_df = df[df["lt_type"].isin(test_qinstances)]
    else:
        random.seed(cfg["seed"])
        qnames = list(set(df["qname"]))
        # split into train / test data
        test_qnames = random.sample(qnames, int(len(qnames)*args.test_size))
        train_qnames = [q for q in qnames if q not in test_qnames]

        train_df = df[df["qname"].isin(train_qnames)]
        test_df = df[df["qname"].isin(test_qnames)]

    return train_df,test_df

def parse_args_any(args):
    pos = []
    named = {}
    key = None
    for arg in args:
        if key:
            if arg.startswith('--'):
                named[key] = True
                key = arg[2:]
            else:
                named[key] = arg
                key = None
        elif arg.startswith('--'):
            key = arg[2:]
        else:
            pos.append(arg)
    if key:
        named[key] = True
    return (pos, named)

def get_alg(alg, cfg):
    if alg == "avg":
        return AvgPredictor()
    elif alg == "nn":
        return LinuxNN(
                cfg = cfg,
                arch = args.arch, hl1 = args.hl1,
                subplan_ests = args.subplan_ests,
                eval_fn_names = args.eval_fns,
                num_conv_layers = args.num_conv_layers,
                final_act = args.final_act,
                use_wandb = args.use_wandb,
                log_transform_y = args.log_transform_y,
                # batch_size = args.batch_size,
                global_feats = args.global_feats,
                # tags = args.tags,
                # seed = args.seed,
                test_size = args.test_size,
                # val_size = args.val_size,
                eval_epoch = args.eval_epoch,
                # logdir = args.logdir,
                num_epochs = args.num_epochs,
                lr = args.lr, weight_decay = args.weight_decay,
                loss_fn_name = args.loss_fn_name)
    else:
        assert False

def eval_alg(alg, loss_funcs, df, sys_logs, samples_type):
    '''
    '''
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    loss_start = time.time()
    alg_name = alg.__str__()
    exp_name = alg.get_exp_name()
    start = time.time()

    ests = alg.test(df, sys_logs)
    # truey = [plan.graph["latency"] for plan in plans]
    truey = df["runtime"].values
    ests = np.array(ests)

    eval_time = round(time.time() - start, 2)
    print("evaluating alg {} took: {} seconds".format(alg_name, eval_time))

    for loss_func in loss_funcs:
        lossarr = loss_func.eval(ests, truey,
                args=args, samples_type=samples_type,
                )
        worst_idx = np.argpartition(lossarr, -4)[-4:]
        print("***Worst runtime preds for: {}***".format(str(loss_func)))
        print("True: ", np.round(truey[worst_idx], 2))
        print("Ests: ", np.round(ests[worst_idx], 2))

        rdir = os.path.join(args.result_dir, exp_name)
        make_dir(rdir)
        resfn = os.path.join(rdir, loss_func.__str__() + ".csv")

        loss_key = "Final-{}-{}-{}".format(str(loss_func),
                                           samples_type,
                                           "mean")
        wandb.run.summary[loss_key] = np.mean(lossarr)

        loss_median = "Final-{}-{}-{}".format(str(loss_func),
                                           samples_type,
                                           "median")
        wandb.run.summary[loss_median] = np.median(lossarr)

        loss_key2 = "Final-{}-{}-{}".format(str(loss_func),
                                           samples_type,
                                           "99p")
        wandb.run.summary[loss_key2] = np.percentile(lossarr, 99)

        print("tags: {}, samples_type: {}, alg: {}, samples: {}, {}: mean: {}, median: {}, 95p: {}, 99p: {}"\
                .format(cfg["tags"],
                    samples_type, alg, len(lossarr),
                    loss_func.__str__(),
                    np.round(np.mean(lossarr),3),
                    np.round(np.median(lossarr),3),
                    np.round(np.percentile(lossarr,95),3),
                    np.round(np.percentile(lossarr,99),3)))

    print("loss computations took: {} seconds".format(time.time()-loss_start))

def read_flags():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=False,
            default="config.yaml", help="")

    parser.add_argument("--factorized_net_embedding_size", "-es", type=int,
            required=False,
            default=None, help="")

    parser.add_argument("--actual_feats", type=int, required=False,
            default=0)
    parser.add_argument("--table_feat", type=int, required=False,
            default=0, help="1/0; add one-hot features for table in query.")
    parser.add_argument("--col_feat", type=int, required=False,
            default=0, help="1/0; add one-hot features for columns in query.")
    parser.add_argument("--feat_subtree_summary", type=int, required=False,
            default=0, help="")
    parser.add_argument("--feat_undirected_edges", type=int, required=False,
            default=0, help="gcn edges both ways / or directed")
    parser.add_argument("--feat_noncumulative_costs", type=int, required=False,
            default=0)

    parser.add_argument("--subplan_ests", type=int, required=False,
            default=0)

    parser.add_argument("--y_normalizer", type=str, required=False,
            default="none", help="none,std,min-max; normalization scheme for target values")
    parser.add_argument("--normalizer", type=str, required=False,
            default="std", help="none,std,min-max; normalization scheme for features.")

    parser.add_argument("--feat_normalization_data", type=str, required=False,
            default="all", help="train,all,wkey; what data to use for normalizing features")
    parser.add_argument("--y_normalization_data", type=str, required=False,
            default="train", help="train,all,wkey")
    parser.add_argument("--final_act", type=str, required=False,
            default="none", help="add a final activation in the models or not.")

    parser.add_argument("--use_wandb", type=int, required=False,
            default=1, help="")
    parser.add_argument("--wandb_tags", type=str, required=False,
            default=None, help="additional tags for wandb logs")

    parser.add_argument("--log_transform_y", type=int, required=False,
            default=0, help="predicting log(latency) instead of latency")

    parser.add_argument("--result_dir", type=str, required=False,
            default="results",
            help="")

    parser.add_argument("--seed", type=int, required=False,
            default=None, help="seed for train/test split")
    parser.add_argument("--num_instances", type=int, required=False,
            default=None, help="seed for train/test split")

    parser.add_argument("--test_size", type=float, required=False,
            default=0.0)

    ## NN parameters
    parser.add_argument("--lr", type=float, required=False,
            default=0.00005)
    parser.add_argument("--weight_decay", type=float, required=False,
            default=0.0)
    parser.add_argument("--hl1", type=int, required=False,
            default=512)
    parser.add_argument("--num_conv_layers", type=int, required=False,
            default=4)
    parser.add_argument("--eval_epoch", type=int, required=False,
            default=2)
    parser.add_argument("--num_epochs", type=int, required=False,
            default=500)

    parser.add_argument("--alg", type=str, required=False,
            default="nn")

    parser.add_argument("--eval_fns", type=str, required=False,
            default="latency_qerr,latency_mse",
            help="final evaluation functions used to evaluate training alg")

    parser.add_argument("--loss_fn_name", type=str, required=False,
            default="mse")

    parser.add_argument("--arch", type=str, required=False,
            default="factorized", help="tcnn/gcn; architecture of trained neural net.")

    parser.add_argument("--global_feats", type=int, required=False,
            default=0)

    parser.add_argument("--skip_timeouts", type=int, required=False,
            default=1)

    return parser.parse_args()

def load_dfs_linux(dirs, tags):
    tags = tags.split(",")
    dirs = dirs.split(",")

    all_dfs = []
    sys_logs = {}

    for curdir in dirs:
        for tag in tags:
            cdf,clogs = load_all_logs_linux(tag, curdir,
                    skip_timeouts=args.skip_timeouts)

            if len(clogs) == 0:
                continue

            # TODO: do this in load_all_logs
            # maxlogtime = max(clogs["timestamp"])
            # try:
                # cdf = cdf[cdf["start_time"] <= maxlogtime]
            # except Exception as e:
                # print(tag, e)
                # continue
            assert tag not in sys_logs

            sys_logs[tag] = clogs
            cdf["tag"] = tag
            all_dfs.append(cdf)

    if len(all_dfs) == 0:
        return [],[]

    return pd.concat(all_dfs), sys_logs

def main():
    global args,cfg

    with open(args.config) as f:
        cfg = yaml.safe_load(f.read())

    wandbcfg = {}
    cargs = vars(args)
    for k,v in cfg.items():
        if isinstance(v, dict):
            for k2,v2 in v.items():
                newkey = k+"_"+k2
                if newkey in cargs and cargs[newkey] is not None:
                    v[k2] = cargs[newkey]
                    v2 = cargs[newkey]
                wandbcfg.update({newkey:v2})
        else:
            if k in cargs and cargs[k] is not None:
                cfg[k] = cargs[k]
                v = cargs[k]
            wandbcfg.update({k:v})

    wandb_tags = ["2a"]
    if args.wandb_tags is not None:
        wandb_tags += args.wandb_tags.split(",")

    wandbcfg.update(vars(args))

    if args.use_wandb:
        wandb.init("learned-latency", config=wandbcfg,
                tags=wandb_tags)

    print(yaml.dump(cfg, default_flow_style=False))

    print("Using tags: ", cfg["tags"].split(","))
    df,sys_logs = load_dfs_linux(cfg["traindata_dir"], cfg["tags"])
    tmp = df[df["status"] != 0]
    tmp = tmp[tmp["status"] != 124]
    df = df[df["status"].isin([0, 124])]
    print(df.groupby("status")["status"].count())
    df = df[df["runtime"] > 1.0]
    pgs = df[df["qname"].str.contains("pgrestore")]
    df = df[~df["qname"].str.contains("pgrestore")]
    ## avoid crashed runs
    pgs = pgs[pgs["runtime"] > 150]
    df = pd.concat([df, pgs])
    #df["runtime"] = df.apply(lambda x: min(x["runtime"], 909.0) , axis=1)
    print("Skipped {} failed jobs".format(len(set(tmp["jobhash"]))))

    jobs = set(df["jobhash"])

    flatdata = defaultdict(list)

    for jh in jobs:
        tmp = df[df["jobhash"] == jh]
        added = False
        added_stats = set()

        for idx,row in tmp.iterrows():
            # if str(row["stat_name"]) == "nan":
            if pd.isnull(row["stat_name"]):
                for k in flatdata:
                    if k in row.keys():
                        flatdata[k].append(row[k])
                    else:
                        flatdata[k].append(None)
                break

            if row["stat_name"] == "LLC-load-misses":
                continue

            if row["stat_name"] not in added_stats:
                flatdata[row["stat_name"]].append(row["value"])
                flatdata[row["stat_name"]+"#"].append(row["value2"])
                added_stats.add(row["stat_name"])

            if not added:
                for k in row.keys():
                    if k not in ["value", "value2", "stat_name", "unit",
                            "util?", "unit2"]:
                        flatdata[k].append(row[k])
                added = True

    df = pd.DataFrame(flatdata)

    train_df, test_df = split_workload(df, cfg)

    print("Train instance types: ", set(train_df["lt_type"]))
    print("Test instance types: ", set(test_df["lt_type"]))
    train_qnames = set(train_df["qname"])
    test_qnames = set(test_df["qname"])

    if cfg["use_eval_tags"]:
        assert False
        ## new envs
        # df,sys_logs2 = load_dfs(cfg["eval_dirs"], cfg["eval_tags"])
        # sys_logs.update(sys_logs2)

        # seendf = df[df["qname"].isin(train_qnames)]
        # unseendf = df[~df["qname"].isin(train_qnames)]

        # new_env_seen_plans = get_plans(seendf)
        # new_env_unseen_plans = get_plans(unseendf)
    else:
        new_env_seen_plans = []
        new_env_unseen_plans = []

    print("Training cmds: {}, Training execs: {},\
Test cmds: {}, Test execs: {}".format(
        len(train_qnames), len(train_df), len(test_qnames),
        len(test_df)))

    # if args.feat_normalization_data == "train"

        # feat_plans = train_plans
    # elif args.feat_normalization_data == "all":
        # feat_plans = train_plans + test_plans + new_env_seen_plans + new_env_unseen_plans

    featurizer = LinuxFeaturizer(train_df,
                            sys_logs,
                            cfg,
                            actual_feats = args.actual_feats,
                            feat_undirected_edges = args.feat_undirected_edges,
                            feat_noncumulative_costs = args.feat_noncumulative_costs,
                            log_transform_y = args.log_transform_y,
                            global_feats=args.global_feats,
                            y_normalization_data=args.y_normalization_data,
                            feat_subtree_summary = args.feat_subtree_summary,
                            normalizer = args.normalizer,
                            y_normalizer = args.y_normalizer,
                            table_feat=args.table_feat,
                            col_feat = args.col_feat)

    alg = get_alg(args.alg, cfg)
    exp_name = alg.get_exp_name()
    rdir = os.path.join(args.result_dir, exp_name)
    make_dir(rdir)
    args_fn = os.path.join(rdir, "args.csv")
    args_dict = vars(args)
    with open(args_fn, 'w') as f:
        w = csv.DictWriter(f, args_dict.keys())
        w.writeheader()
        w.writerow(args_dict)

    eval_fns = []
    eval_fn_names = args.eval_fns.split(",")
    for l in eval_fn_names:
        eval_fns.append(get_eval_fn(l))

    alg.train(train_df,
            sys_logs,
            featurizer,
            test = test_df,
            new_env_seen= [],
            new_env_unseen = [],
            )

    eval_alg(alg, eval_fns, train_df, sys_logs, "train")

    if len(test_df) > 0:
        eval_alg(alg, eval_fns, test_df, sys_logs, "test")

    # if len(new_env_seen_plans) > 0:
        # eval_alg(alg, eval_fns, new_env_seen_plans, sys_logs, "new_env_seen")

    # if len(new_env_unseen_plans) > 0:
        # eval_alg(alg, eval_fns, new_env_unseen_plans, sys_logs, "new_env_unseen")

if __name__ == "__main__":
    # mp.set_start_method('spawn')
    cfg = {}
    args = read_flags()
    main()
