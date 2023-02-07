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

import torch
from torch.autograd import Variable

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
            if "Time" in k:
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
                assert False

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

    logfns = glob.iglob(inp_dir + "/*/results/sar_logs*")
    logdfs = {}

    for fi, fn in enumerate(logfns):
        #print(fn)
        if ".csv" in fn or "sar_logs00" in fn:
            continue
        try:
            curdf = pd.read_csv(fn, delimiter=";")
        except Exception as e:
            print(e)
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

def get_plans(df):
    plans = []
    for i,row in df.iterrows():
        exp = row["exp_analyze"]
        try:
            plan = eval(str(exp))
        except Exception as e:
            print(e)
            continue

        G = explain_to_nx(plan[0][0][0])

        # global properties about plan
        G.graph["start_time"] = row["start_time"]
        G.graph["latency"] = row["runtime"]
        G.graph["tag"] = row["tag"]
        G.graph["qname"] = row["qname"]
        plans.append(G)

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


def load_all_logs(inp_tag, inp_dir):

    inp_dir = os.path.join(inp_dir, inp_tag)
    if not os.path.exists(inp_dir):
        return [],[]

    rtfns = glob.iglob(inp_dir + "/*/results/Runtime*.csv")
    dfs = []

    for rtfn in rtfns:
        dfs.append(pd.read_csv(rtfn))

    df = pd.concat(dfs)

    ### TODO: handle timeouts;
    ### just re-use plans from another run?
    df = df[~df["exp_analyze"].isna()]
    df = df[df["exp_analyze"] != "nan"]

    df["inp_tag"] = inp_tag

    logdfs = load_sys_logs(inp_dir)
    if len(logdfs) == 0:
        return df,[]

    logdfs["inp_tag"] = inp_tag

    return df, logdfs



def to_variable(arr, use_cuda=True, requires_grad=False):
    if isinstance(arr, list) or isinstance(arr, tuple):
        arr = np.array(arr)
    if isinstance(arr, np.ndarray):
        arr = Variable(torch.from_numpy(arr), requires_grad=requires_grad)
    else:
        arr = Variable(arr, requires_grad=requires_grad)

    return arr

