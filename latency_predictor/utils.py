import pandas as pd
import numpy as np
import seaborn as sns
import glob
from collections import defaultdict
import pdb
import xmltodict
import networkx as nx
import re
import hashlib
import os
import errno
import time
from networkx.drawing.nx_agraph import write_dot,graphviz_layout
from networkx.algorithms.traversal.depth_first_search import dfs_tree
import math
from io import StringIO

import torch
from torch.autograd import Variable

MIN_EST = 0.1
# MIN_EST = 1.0
MAX_EST = 909.0

ALL_INSTANCES = ['a1_large_gp3_4g',
 'c5a_large_mag_4g',
 'c7g_large_mag_4g',
 'm6a_large_mag_8g',
 'r6a_large_mag_16g',
 'r7g_large_gp2_16g',
 't3_large_gp2_8g',
 't3_xlarge_gp2_16g',
 't3a_medium_gp3_4g',
 't4g_large_mag_8g',
 't7xlarge-gp3-d'
 ]

# USE_TEST_INSTANCES = True
USE_TEST_INSTANCES = False
TEST_INSTANCE_TYPES = ["a1_large_gp3_4g", "r7g_large_gp2_16g",
        "t3a_medium_gp3_4g",
        "m6a_large_mag_8g",
        #"t4g_large_mag_8g",
        "c7g_large_mag_4g",
        "t3_large_gp2_8g",
        ]

RUNTIME_MASK = [0, 1,
                0, 1,
                1]

def explain_to_nx(plan):
    def recurse(G, cur_plan, cur_key, cur_node):
        if cur_plan is None:
            return

        #if isinstance(cur_plan, dict):
        new_node = cur_plan["Node Type"] + str(hash(str(cur_plan)))[0:6]
        G.add_node(new_node)
        if cur_node is not None:
            G.add_edge(cur_node,
                      new_node)

        # print(cur_plan.keys())
        for k,v in cur_plan.items():
            if k == "Plans":
                continue
            if isinstance(v, list) or isinstance(v, dict):
                v = str(v)

            k = k.replace(" ", "")
            if "Time" in k and is_float(v):
                G.nodes()[new_node][k] = v / 1000.0
            else:
                G.nodes()[new_node][k] = v

        if "Actual Total Time" in cur_plan:
            if "Plans" not in cur_plan:
                children_time = 0.0
            elif len(cur_plan["Plans"]) == 2:
                children_time = cur_plan["Plans"][0]["Actual Total Time"] \
                        + cur_plan["Plans"][1]["Actual Total Time"]
            elif len(cur_plan["Plans"]) == 1:
                children_time = cur_plan["Plans"][0]["Actual Total Time"]
            else:
                return
                # assert False

            # print(children_time)
            G.nodes[new_node]["ActualCurTime"] =(cur_plan["Actual Total Time"]\
                    -children_time) / 1000

        if "Plans" in cur_plan:
            for nextp in cur_plan["Plans"]:
                recurse(G, nextp, "", new_node)

    attrs = defaultdict(set)
    G = nx.DiGraph()
    recurse(G, plan["Plan"], "", None)

    # assert nx.is_tree(G)
    if not nx.is_tree(G):
        print("not tree")
        # print(G.nodes())
        # pdb.set_trace()

    return G

def extract_values(obj, key):
    """Recursively pull values of specified key from nested JSON."""
    arr = []

    def extract(obj, arr, key):
        """Return all matching values in an object."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, key)
                elif k == key:
                    # if "Scan" in v:
                        # print(v)
                        # pdb.set_trace()
                    # if "Join" in v:
                        # print(obj)
                        # pdb.set_trace()
                    arr.append(v)

        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    results = extract(obj, arr, key)
    return results

def is_float(val):
    try:
        float(val)
        return True
    except:
        return False

def get_sub_distables(G, node):
    if "@DistTables" not in G.nodes()[node]:
        return []

    curtables = G.nodes()[node]["@DistTables"].split(",")

    for edge in G.out_edges(node):
        curtables += get_sub_distables(G, edge[1])

    curtables = list(set(curtables))
    curtables.sort()

    return curtables

def get_subrows(G, node):
    # if this node has a cost, then return that.
    # if this node has 0 cost, then find the subcost of all children
    cest = float(G.nodes()[node]["@EstimateRows"])
    if cest != 0:
        return cest

    total_rows = 0.0
    for edge in G.out_edges(node):
        total_rows += get_subrows(G, edge[1])

    return total_rows

def get_subcost(G, node):
    # if this node has a cost, then return that.
    # if this node has 0 cost, then find the subcost of all children
    cest = float(G.nodes()[node]["@PDWAccumulativeCost"])
    if cest != 0:
        return cest

    total_cost = 0.0
    for edge in G.out_edges(node):
        total_cost += get_subcost(G, edge[1])

    return total_cost

def get_curcost(G, node):
    cest = float(G.nodes()[node]["@PDWAccumulativeCost"])

    if cest == 0:
        return 0

    past_cost = 0.0
    for edge in G.out_edges(node):
        assert edge[0] == node
        past_cost += get_subcost(G, edge[1])

    return cest - past_cost

def deterministic_hash(string):
    return int(hashlib.sha1(str(string).encode("utf-8")).hexdigest(), 16)

def get_int_from_str(txt):
    num = int(re.search(r'\d+', txt).group(0))
    return num

def make_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def get_all_traindirs(inp_dir):
    '''
    goes through all the log files in inp_dir, and loads the pd dataframe with
    appropriate latencies, and costs.
    '''
    pass

def load_sys_logs(inp_dir):

    # logfns = glob.iglob(inp_dir + "/*/results/sar_logs*")
    logfns = glob.iglob(inp_dir + "/results/sar_logs*")
    logdfs = {}

    for fi, fn in enumerate(logfns):
        #print(fn)
        if ".csv" in fn or "sar_logs00" in fn:
            continue
        try:
            curdf = pd.read_csv(fn, delimiter=";")
        except Exception as e:
            continue
        if len(curdf) == 0:
            continue

        ts_reps = curdf.groupby("timestamp")["interval"].count()\
                .reset_index().sort_values(by="interval",
                ascending=False)["interval"].values[0]

        if ts_reps > 1:
            curdf = curdf.groupby("timestamp").mean().reset_index()
            curdf = curdf.drop(columns=["interval"])
        else:
            curdf = curdf.drop(columns=["# hostname", "interval"])

        if 'kbmemfree' in curdf.keys():
            logdfs["mem"] = curdf
        elif "dropd/s" in curdf.keys():
            logdfs["network1"] = curdf
        elif "%sio-10" in curdf.keys():
            #system load and pressure-stall statistics
            logdfs["pressure_io"] = curdf
        elif 'kbhugfree' in curdf.keys():
            logdfs["hugepg"] = curdf
        elif '%smem-10' in curdf.keys():
            logdfs["pressure_mem"] = curdf
        elif 'kbswpfree' in curdf.keys():
            logdfs["swap"] = curdf
        elif 'idgm6/s' in curdf.keys():
            logdfs["network2"] = curdf
        elif 'bdscd/s' in curdf.keys():
            logdfs["io1"] = curdf
        elif '%irq' in curdf.keys():
            logdfs["cpu_utilization"] = curdf
        elif 'atmptf/s' in curdf.keys():
            logdfs["network3"] = curdf
        elif 'idgm/s' in curdf.keys():
            logdfs["network4"] = curdf
        elif 'areq-sz' in curdf.keys():
            logdfs["device_io"] = curdf
        elif 'call/s' in curdf.keys():
            logdfs["network_nfs"] = curdf
        elif 'rxdrop/s' in curdf.keys():
            logdfs["network5"] = curdf
        elif 'rxcmp/s' in curdf.keys():
            logdfs["network6"] = curdf
        elif 'tcp6sck' in curdf.keys():
            logdfs["network7"] = curdf
        elif 'iseg/s' in curdf.keys():
            logdfs["network8"] = curdf
        elif 'pswpin/s' in curdf.keys():
            logdfs["swap"] = curdf
        elif 'intr/s' in curdf.keys():
            logdfs["interrupts"] = curdf
        elif '%scpu-10' in curdf.keys():
            logdfs["pressure_cpu"] = curdf
        elif 'MHz' in curdf.keys():
            logdfs["power-cpu"] = curdf
        elif 'degC' in curdf.keys():
            logdfs["power-temp"] = curdf
        elif '%ufsused' in curdf.keys():
            logdfs["filesystem"] = curdf
        elif 'cswch/s' in curdf.keys():
            logdfs["context_switch"] = curdf
        elif 'pgpgin/s' in curdf.keys():
            logdfs["paging"] = curdf
        elif 'fwddgm/s' in curdf.keys():
            logdfs["network9"] = curdf
        elif 'ihdrer6/s' in curdf.keys():
            logdfs["network10"] = curdf
        elif 'imsg/s' in curdf.keys():
            logdfs["network11"] = curdf
        elif 'ierr6/s' in curdf.keys():
            logdfs["network12"] = curdf
        elif 'scall/s' in curdf.keys():
            logdfs["network13"] = curdf
        elif 'runq-sz' in curdf.keys():
            logdfs["pressure_load"] = curdf
        elif 'dentunusd' in curdf.keys():
            logdfs["inode"] = curdf
        elif 'igmbq6/s' in curdf.keys():
            logdfs["network14"] = curdf
        elif 'tcpsck' in curdf.keys():
            logdfs["network15"] = curdf
        elif 'otmex/s' in curdf.keys():
            logdfs["network16"] = curdf
        elif 'ihdrerr/s' in curdf.keys():
            logdfs["network17"] = curdf
        elif 'irec6/s' in curdf.keys():
            logdfs["network18"] = curdf
        else:
            assert False

    keys = list(logdfs.keys())
    for k in keys:
        if "network" in k:
            del logdfs[k]
            continue
        if k in ["power-cpu", "hugepg", "power-temp"]:
            del logdfs[k]
            continue

    keys = list(logdfs.keys())
    if len(keys) == 0:
        return []

    curdf = logdfs[keys[0]]

    for ki in range(1,len(keys)):
        curdf = curdf.merge(logdfs[keys[ki]], on="timestamp")

    return curdf

def num_concurrent(row, df):
    tmp = df[(df["start_time"] <= row["start_time"]) &
            (df["end_time"] >= row["start_time"])]
    return len(tmp)


def get_plans(df):
    '''
    We want plans to be sorted by <instance, timestamp>.
    '''
    import pickle
    FN = "baselines/linear_models.pkl"
    if os.path.exists(FN):
        with open(FN, "rb") as file:
            loaded_models = pickle.load(file)
    else:
        loaded_models = None

    plans = []
    instances = set(df["instance"])
    for instance in instances:
        tmp = df[df["instance"] == instance]
        tmp = tmp.sort_values(by="start_time",
                ascending=True)
        # tmp["end_time"] = tmp.apply(lambda x: x["start_time"] + x["runtime"] ,
                # axis=1)
        # start = time.time()
        # tmp["num_concurrent"] = tmp.apply(lambda x: num_concurrent(x, tmp),
                    # axis=1)
        # if max(tmp["num_concurrent"]) > 1:
            # print(set(tmp["tag"]))
            # print("num concurrent took: ", time.time()-start)
            # print(tmp["num_concurrent"].describe())
            # pdb.set_trace()

        for i,row in tmp.iterrows():
            exp = row["exp_analyze"]
            try:
                plan = eval(str(exp))
            except Exception as e:
                # print(e)
                continue

            G = explain_to_nx(plan[0][0][0])

            # global properties about plan
            G.graph["start_time"] = row["start_time"]
            G.graph["latency"] = row["runtime"]
            G.graph["tag"] = row["tag"]
            G.graph["qname"] = row["qname"]
            G.graph["instance"] = row["instance"]
            G.graph["lt_type"] = row["lt_type"]
            G.graph["bk_kind"] = row["bk_kind"]
            G.graph["concurrent"] = row["concurrent"]

            pdata = dict(G.nodes(data=True))
            max_cost = max(subdict['TotalCost'] for subdict in
                    pdata.values())

            # G.graph["heuristic_cost"] = max_cost
            ## to ignore the cost
            G.graph["heuristic_cost"] = 1.0

            if loaded_models is not None and \
                    row["lt_type"] in loaded_models:
                model = loaded_models[row["lt_type"]]
                X_reshaped = np.array([max_cost]).reshape(-1, 1)
                prediction = model.predict(X_reshaped)
                G.graph["heuristic_pred"] = prediction[0]
            else:
                # assert False
                # print("no model for: ", row["lt_type"])
                G.graph["heuristic_pred"] = 0.0

            # 1d rep
            pdata = dict(G.nodes(data=True))
            actual_card = sum(subdict['ActualRows'] for subdict in
                    pdata.values())
            est_card = sum(subdict['PlanRows'] for subdict in
                    pdata.values())
            actual_time = max(subdict['ActualTotalTime'] for subdict in
                    pdata.values())
            est_cost = max(subdict['TotalCost'] for subdict in
                    pdata.values())
            pred_time = G.graph["heuristic_pred"]

            G.graph["runtime_feats"] = [actual_card, est_card,
                                    actual_time, pred_time,
                                    est_cost]
            plans.append(G)

    # for i,row in df.iterrows():
        # exp = row["exp_analyze"]
        # try:
            # plan = eval(str(exp))
        # except Exception as e:
            # # print(e)
            # continue

        # G = explain_to_nx(plan[0][0][0])

        # # global properties about plan
        # G.graph["start_time"] = row["start_time"]
        # G.graph["latency"] = row["runtime"]
        # G.graph["tag"] = row["tag"]
        # G.graph["qname"] = row["qname"]
        # G.graph["instance"] = row["instance"]
        # G.graph["lt_type"] = row["lt_type"]
        # G.graph["bk_kind"] = row["bk_kind"]
        # plans.append(G)

    return plans

def extract_previous_logs(cur_sys_logs, start_time,
                        prev_secs,
                        skip_logs,
                        max_log_len,
                        ):

    tmp = cur_sys_logs[(cur_sys_logs["timestamp"] <= start_time) & \
            (cur_sys_logs["timestamp"] >= start_time - prev_secs)]
    tmp = tmp.iloc[::skip_logs, :]
    tmp = tmp.tail(max_log_len)

    return tmp

LT_FN = "lt_instances.txt"
## ignoring these LT_TYPES for noe
OTHERS = '''
a1_large_mag_4g=lt-0d15fb8f5bbe9a27d
t3a_large_gp3_8g=lt-084bfbae110d52d4e
r7g_medium_gp2_16g=lt-01d0081183a7d79f2
'''

LT_TYPES = '''
a1_large_gp3_4g=lt-04840b55d3f795395
r7g_large_gp2_16g=lt-0212ec953ba35b176
t3_large_gp2_8g=lt-05d2d354bc3dd9133
c5a_large_mag_4g=lt-03218e9e27718bbbe
m6a_large_mag_8g=lt-0f6f46002652f9a4c
t3a_medium_gp3_4g=lt-0af65294350b1a8c1
r6a_large_mag_16g=lt-0e608666ff3adff07
t4g_large_mag_8g=lt-04e0b4826c63bfadb
c7g_large_mag_4g=lt-0af47c6caa3b53b8b
t3_xlarge_gp2_16g=lt-0b413bcc22b3ac8fb
'''

LT_TYPES_DF = pd.read_csv(StringIO(LT_TYPES), sep="=", header=None,
                       names=["lt_type", "lt"])

def get_bk_kind(bdf, start_time, runtime):
    tmp = bdf[bdf["timestamp"] < start_time]
    tmp = tmp[tmp["end_timestamp"] > start_time]
    if len(tmp) == 0:
        #print(tmp)
        return "None"
    #tmp = tmp[tmp["timestamp"] == max(tmp["timestamp"])]
    #assert len(tmp) <= 1

    if len(tmp) == 0:
        return "None"
    vals = list(tmp["kind"].values)
    vals.sort()
    return "-".join(vals)

def get_bk_concurrent(bdf, start_time, runtime):
    tmp = bdf[(bdf["timestamp"] < start_time) & \
                (bdf["end_timestamp"] > start_time)]
    # tmp = tmp[tmp["end_timestamp"] > start_time]
    return len(tmp)
    # if len(tmp) == 0:
        # #print(tmp)
        # return 0
    # #tmp = tmp[tmp["timestamp"] == max(tmp["timestamp"])]
    # #assert len(tmp) <= 1

    # if len(tmp) == 0:
        # return 0
    # vals = list(tmp["kind"].values)
    # vals.sort()
    # return "-".join(vals)

def load_all_logs(inp_tag, inp_dir, skip_timeouts=False):

    inp_dir = os.path.join(inp_dir, inp_tag)
    if not os.path.exists(inp_dir):
        return [],[]

    dfs = []
    all_logs = {}

    instance_dirs = os.listdir(inp_dir)
    lt_fn_path = os.path.join(inp_dir, LT_FN)
    if os.path.exists(lt_fn_path):
        ltdf = pd.read_csv(lt_fn_path, header=None,
                   names=["instance", "lt"])
        # excluding some instance types
        ltdf = ltdf[ltdf["lt"].isin(LT_TYPES_DF["lt"].values)]
        ltdf = ltdf.merge(LT_TYPES_DF, on="lt")
    else:
        return [],[]

    for iname in instance_dirs:
        curdir = os.path.join(inp_dir, iname)

        if not os.path.isdir(curdir):
            continue
        # lets load the result files
        rtfns = glob.iglob(curdir + "/results/Runtime*.csv")
        curdfs = []

        for rtfn in rtfns:
            try:
                currt = pd.read_csv(rtfn,
                        # skipfooter=1,
                        # engine="python",
                        error_bad_lines=False)
            except Exception as e:
                print(e)
                continue

            currt["instance"] = iname
            curdfs.append(currt)

        if len(curdfs) == 0:
            continue

        currt = pd.concat(curdfs)
        if len(currt) == 0:
            continue
        currt = currt.merge(ltdf, on="instance")

        if len(currt) == 0:
            # print(iname, "rt == 0")
            continue

        curlogs = load_sys_logs(curdir)

        # skip queries which are outside logged time
        if len(curlogs) == 0:
            continue

        curlogs = curlogs.dropna()
        if len(curlogs) == 0:
            continue

        # remove part of logs which aren't in currts
        max_query_start = max(currt["start_time"].values) + 60.0
        min_query_start = min(currt["start_time"].values) - 650.0

        curlogs = curlogs[curlogs["timestamp"] > min_query_start]
        curlogs = curlogs[curlogs["timestamp"] < max_query_start]

        if len(curlogs) == 0:
            print("curlogs == 0 after filtering for: ", inp_tag)
            continue

        maxlogtime = max(curlogs["timestamp"].values)
        try:
            currt = currt[currt["start_time"] <= maxlogtime]
        except Exception as e:
            # print(e)
            continue

        if len(currt) == 0:
            continue

        bkfns = glob.iglob(curdir + "/results/Background*.csv")
        bdfs = []
        for bkfn in bkfns:
            instance_name = os.path.basename(os.path.dirname(os.path.dirname(bkfn)))
            try:
                bk = pd.read_csv(bkfn, error_bad_lines=False, skipfooter=1,
                        engine="python")
            except:
                continue

            bk["instance"] = instance_name
            bk = bk.merge(ltdf, on="instance")
            if len(bk) == 0:
                continue
            bdfs.append(bk)

        if len(bdfs) != 0:
            bdf = pd.concat(bdfs)
            currt["bk_kind"] = currt.apply(lambda x:
                    get_bk_kind(bdf, x["start_time"], x["runtime"]) , axis=1)
            currt["bk_concurrent"] = currt.apply(lambda x:
                    get_bk_concurrent(bdf, x["start_time"], x["runtime"]),
                    axis=1)
        else:
            currt["bk_kind"] = "None"
            currt["bk_concurrent"] = 1.0

        if "new" in curdir:
            concurrent=False
        else:
            concurrent=True

        currt["concurrent"]  = concurrent

        # print("Background workloads: ", set(currt["bk_kind"]))

        # final collection
        dfs.append(currt)
        all_logs[iname] = curlogs

    if len(dfs) == 0:
        return [],[]

    df = pd.concat(dfs)

    if len(all_logs) == 0:
        return [],[]

    # handle timeouts
    timeouts = df[df["exp_analyze"].isna()]
    timeouts = timeouts[timeouts["runtime"] != -1.0]
    df = df[~df["exp_analyze"].isna()]
    df = df[df["exp_analyze"] != "nan"]
    df = df[df["runtime"] >= 0.0]

    if not skip_timeouts and len(timeouts) != 0:
        print(inp_dir, inp_tag)
        print("Total: ", len(df), "Timeouts: ", len(timeouts))
        timeouts = timeouts[timeouts["qname"].isin(df["qname"].values)]
        timeouts["exp_analyze"] = timeouts.apply(lambda x:
                df[df["qname"] == x["qname"]]["exp_analyze"].values[0],
                axis=1)
        df = pd.concat([df, timeouts])

    return df, all_logs

PERF_NAMES = ["value", "unit", "stat_name", "job_time", "util?",
        "value2", "unit2"]

def load_all_logs_linux(inp_tag, inp_dir,
        skip_timeouts=False):
    inp_dir = os.path.join(inp_dir, inp_tag)
    if not os.path.exists(inp_dir):
        print("no exists")
        return [],[]

    dfs = []
    all_logs = {}

    instance_dirs = os.listdir(inp_dir)
    lt_fn_path = os.path.join(inp_dir, LT_FN)
    if os.path.exists(lt_fn_path):
        ltdf = pd.read_csv(lt_fn_path, header=None,
                   names=["instance", "lt"])
        ltdf = ltdf.merge(LT_TYPES_DF, on="lt")
    else:
        print("lt fn path doesn't exist")
        return [],[]

    for iname in instance_dirs:
        curdir = os.path.join(inp_dir, iname)
        if not os.path.isdir(curdir):
            continue

        perfcsvs = glob.iglob(curdir + "/results/perf/*.csv")
        cmds_fn = curdir + "/results/perf/allcommands.csv"

        cmdsdf = pd.read_csv(cmds_fn,
                        names=["jobhash", "cmd", "fn", "start_time","runtime", "status"],
                error_bad_lines=False,
                header=None,
                skipfooter=1,
                )
        cmdsdf["instance"] = iname
        curdfs = []

        for pfn in perfcsvs:
            hashcsv = os.path.basename(pfn)
            jobhash = hashcsv.replace(".csv", "")

            if "allcommands" in pfn:
                continue
            try:
                currt = pd.read_csv(pfn, skiprows=1,
                                    names = PERF_NAMES,
                                    skipfooter=1,
                                    header=None)
                #print(currt)
            except Exception as e:
                # print(e)
                continue
            if len(currt) == 0:
                continue

            currt["jobhash"] = jobhash
            #currt["instance"] = iname
            curdfs.append(currt)

        if len(curdfs) == 0:
            continue

        currt = pd.concat(curdfs)
        #currt = currt[currt["jobhash"].isin(cmdsdf["jobhash"].values)]
        if len(currt) == 0:
            continue

        cmdsdf = cmdsdf.merge(ltdf, on="instance")
        currt = currt.merge(cmdsdf, on="jobhash", how="right")

        if len(currt) == 0:
            print(iname, "rt == 0")
            continue

        curlogs = load_sys_logs(curdir)

        # skip queries which are outside logged time
        if len(curlogs) == 0:
            continue

        curlogs = curlogs.dropna()
        if len(curlogs) == 0:
            continue

        # remove part of logs which aren't in currts
        # max_query_start = max(currt["start_time"].values) + 60.0
        # min_query_start = min(currt["start_time"].values) - 650.0
        # curlogs = curlogs[curlogs["timestamp"] > min_query_start]
        # curlogs = curlogs[curlogs["timestamp"] < max_query_start]

        if len(curlogs) == 0:
            print("curlogs == 0 after filtering for: ", inp_tag)
            continue

        if len(currt) == 0:
            continue

        dfs.append(currt)
        all_logs[iname] = curlogs

    if len(dfs) == 0:
        return [],[]

    df = pd.concat(dfs)

    if len(all_logs) == 0:
        return [],[]

    df["qname"] = df.apply(lambda x: x["fn"]+x["cmd"] ,
            axis=1)

    if min(df["start_time"]) > max(curlogs["timestamp"]):
        df["start_time"] = df.apply(lambda x: x["start_time"] / 1000000000.0,
                axis=1)
        print("updated nanoseconds to seconds!")

    return df, all_logs

def to_variable(arr, use_cuda=True, requires_grad=False):
    if isinstance(arr, list) or isinstance(arr, tuple):
        arr = np.array(arr)
    if isinstance(arr, np.ndarray):
        arr = Variable(torch.from_numpy(arr), requires_grad=requires_grad)
    else:
        arr = Variable(arr, requires_grad=requires_grad)

    return arr

