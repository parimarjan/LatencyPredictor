import networkx as nx
# import xgboost as xgb
import wandb
import random
import numpy as np
from collections import defaultdict
import pdb

class LatencyPredictor():

    def __init__(self, *args, **kwargs):
        # TODO: set each of the kwargs as variables
        pass

    def train(self, train_plans, sys_logs, featurizer,
            **kwargs):
        pass

    def test(self, plans, sys_logs, **kwargs):
        '''
        @ret: [latencies].
        '''
        pass

    def get_exp_name(self):
        name = self.__str__()
        # if self.rand_id is None:
        if not hasattr(self, "rand_id"):
            self.rand_id = wandb.util.generate_id()

        name += self.rand_id
        return name

    def num_parameters(self):
        '''
        size of the parameters needed so we can compare across different algorithms.
        '''
        return 0

    def __str__(self):
        return self.__class__.__name__

    def save_model(self, save_dir="./", suffix_name=""):
        pass

class AvgPredictor(LatencyPredictor):

    def __init__(self, *args, **kwargs):
        # TODO: set each of the kwargs as variables
        pass

    def test(self, plans, sys_logs, **kwargs):
        '''
        '''
        ret = []
        qtimes = defaultdict(list)
        for plan in plans:
            qtimes[plan.graph["qname"]].append(plan.graph["latency"])

        for plan in plans:
            ret.append(np.mean(qtimes[plan.graph["qname"]]))

        return ret
