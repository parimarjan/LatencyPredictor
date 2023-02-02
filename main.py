import argparse
import random
import time
import ntpath
import os
import pdb

from latency_predictor.utils import *
from latency_predictor.algs import *
from latency_predictor.nn import NN
from latency_predictor.eval_fns import *
from latency_predictor.featurizer import *
import wandb
import logging
import csv

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)

def get_alg(alg):
    if alg == "nn":
        return NN(arch = args.arch, hl1 = args.hl1,
                subplan_ests = args.subplan_ests,
                eval_fn_names = args.eval_fns,
                num_conv_layers = args.num_conv_layers,
                final_act = args.final_act,
                use_wandb = args.use_wandb,
                log_transform_y = args.log_transform_y,
                batch_size = args.batch_size,
                global_feats = args.global_feats,
                tags = args.tags,
                seed = args.seed, test_size = args.test_size,
                val_size = args.val_size,
                eval_epoch = args.eval_epoch,
                # logdir = args.logdir,
                num_epochs = args.num_epochs,
                lr = args.lr, weight_decay = args.weight_decay,
                loss_fn_name = args.loss_fn_name)
    else:
        assert False

def eval_alg(alg, loss_funcs, plans, sys_logs, samples_type):
    '''
    '''
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    loss_start = time.time()
    alg_name = alg.__str__()
    exp_name = alg.get_exp_name()
    start = time.time()

    ests = alg.test(plans, sys_logs)
    truey = [plan.graph["latency"] for plan in plans]

    eval_time = round(time.time() - start, 2)
    print("evaluating alg {} took: {} seconds".format(alg_name, eval_time))

    for loss_func in loss_funcs:
        lossarr = loss_func.eval(ests, truey,
                args=args, samples_type=samples_type)

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
                .format(args.tags,
                    samples_type, alg, len(lossarr),
                    loss_func.__str__(),
                    np.round(np.mean(lossarr),3),
                    np.round(np.median(lossarr),3),
                    np.round(np.percentile(lossarr,95),3),
                    np.round(np.percentile(lossarr,99),3)))

    print("loss computations took: {} seconds".format(time.time()-loss_start))

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_true_rows", type=int, required=False,
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
            default="min-max", help="none,std,min-max; normalization scheme for features.")

    parser.add_argument("--feat_normalization_data", type=str, required=False,
            default="train", help="train,all,wkey; what data to use for normalizing features")
    parser.add_argument("--y_normalization_data", type=str, required=False,
            default="train", help="train,all,wkey")
    parser.add_argument("--final_act", type=str, required=False,
            default="none", help="add a final activation in the models or not.")

    parser.add_argument("--tags", type=str, required=False,
            default="t7xlarge-gp3-d",
            help="tags to use for training data")
    parser.add_argument("--use_eval_tags", type=int, required=False,
            default=0, help="0 or 1. use additional tags for evaluating models or not.")

    parser.add_argument("--eval_tags", type=str, required=False,
            default=None, help="additional tag to use for evaluating the models. Leave None to use all possible tags")

    parser.add_argument("--use_wandb", type=int, required=False,
            default=1, help="")
    parser.add_argument("--wandb_tags", type=str, required=False,
            default=None, help="additional tags for wandb logs")

    parser.add_argument("--batch_size", type=int, required=False,
            default=1, help="""batch size for training gcn models.""")

    parser.add_argument("--log_transform_y", type=int, required=False,
            default=0, help="predicting log(latency) instead of latency")

    parser.add_argument("--traindata_dir", type=str, required=False,
            default="""LatencyCollectorResults/multiple""", help="directory with the workload log files.")

    parser.add_argument("--result_dir", type=str, required=False,
            default="results",
            help="")

    parser.add_argument("--seed", type=int, required=False,
            default=666, help="seed for train/test split")

    parser.add_argument("--test_size", type=float, required=False,
            default=0.2)
    parser.add_argument("--val_size", type=float, required=False,
            default=0.2)

    ## NN parameters
    parser.add_argument("--lr", type=float, required=False,
            default=0.0001)
    parser.add_argument("--weight_decay", type=float, required=False,
            default=0.0)
    parser.add_argument("--hl1", type=int, required=False,
            default=256)
    parser.add_argument("--num_conv_layers", type=int, required=False,
            default=4)
    parser.add_argument("--eval_epoch", type=int, required=False,
            default=100)
    parser.add_argument("--num_epochs", type=int, required=False,
            default=10)

    parser.add_argument("--alg", type=str, required=False,
            default="nn")

    parser.add_argument("--eval_fns", type=str, required=False,
            default="latency_qerr,latency_mse",
            help="final evaluation functions used to evaluate training alg")

    parser.add_argument("--loss_fn_name", type=str, required=False,
            default="mse")

    parser.add_argument("--arch", type=str, required=False,
            default="gcn", help="tcnn/gcn; architecture of trained neural net.")

    parser.add_argument("--global_feats", type=int, required=False,
            default=0)

    return parser.parse_args()

def main():
    global args

    if args.use_wandb:
        wandb_tags = ["1a"]
        if args.wandb_tags is not None:
            wandb_tags += args.wandb_tags.split(",")

        wandb.init("learned-latency", config={},
                tags=wandb_tags)
        wandb.config.update(vars(args))

    tags = args.tags.split(",")
    print("Using tags: ", tags)

    all_dfs = []
    sys_logs = {}
    qnames = set()

    for tag in tags:
        cdf,clogs = load_all_logs(tag, args.traindata_dir)
        sys_logs[tag] = clogs
        cdf["tag"] = tag

        all_dfs.append(cdf)
        qnames = qnames.union(set(cdf["qname"]))

    random.seed(args.seed)
    # split into train / test data
    test_qnames = random.sample(qnames, int(len(qnames)*args.test_size))
    train_qnames = [q for q in qnames if q not in test_qnames]

    train_dfs,test_dfs,train_plans,test_plans = [],[],[],[]
    train_samples,test_samples = 0,0

    for df in all_dfs:
        train_dfs.append(df[df["qname"].isin(train_qnames)])
        # train_plans.append(get_plans(train_dfs[-1]))
        train_plans += get_plans(train_dfs[-1])
        test_dfs.append(df[df["qname"].isin(test_qnames)])
        # test_plans.append(get_plans(test_dfs[-1]))
        test_plans += get_plans(test_dfs[-1])

        train_samples += len(train_dfs[-1])
        test_samples += len(test_dfs[-1])

    new_env_seen = None
    new_env_unseen = None

    print("Num training queries: {}, Num training samples: {},\
Num test queries: {}, Num test samples: {}".format(
        len(train_qnames), train_samples, len(test_qnames),
        test_samples))

    featurizer = Featurizer(train_plans,
                            # use_true_rows = args.use_true_rows,
                            feat_undirected_edges = args.feat_undirected_edges,
                            feat_noncumulative_costs = args.feat_noncumulative_costs,
                            log_transform_y = args.log_transform_y,
                            global_feats=args.global_feats,
                            # feat_normalization_data=args.feat_normalization_data,
                            y_normalization_data=args.y_normalization_data,
                            feat_subtree_summary = args.feat_subtree_summary,
                            normalizer = args.normalizer,
                            y_normalizer = args.y_normalizer,
                            table_feat=args.table_feat,
                            col_feat = args.col_feat,
                            num_bins=50)

    alg = get_alg(args.alg)
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

    alg.train(train_plans,
            sys_logs,
            featurizer,
            new_env_seen= new_env_seen,
            new_env_unseen = new_env_unseen
            )

    eval_alg(alg, eval_fns, train_plans, sys_logs, "train")

    # pdb.set_trace()


    # alg.train(traindata, valdata=valdata, testdata=testdata,
            # seenevaldata=seeneval,
            # unseenevaldata=unseeneval,
            # featurizer=featurizer)
    # eval_alg(alg, eval_fns, traindata, "train")

    # if len(valdata["lats"]) > 0:
        # eval_alg(alg, eval_fns, valdata, "val")
    # if len(testdata["lats"]) > 0:
        # eval_alg(alg, eval_fns, testdata, "test")

    # if seeneval is not None:
        # eval_alg(alg, eval_fns, seeneval, "seeneval")

    # if unseeneval is not None:
        # eval_alg(alg, eval_fns, unseeneval, "unseeneval")

if __name__ == "__main__":
    args = read_flags()
    main()
