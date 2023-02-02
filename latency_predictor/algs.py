import networkx as nx
# import xgboost as xgb
import wandb
import random
import numpy as np
import pdb

class LatencyPredictor():

    def __init__(self, *args, **kwargs):
        # TODO: set each of the kwargs as variables
        pass

    def train(self, traindata, **kwargs):
        pass

    def test(self, testdata, **kwargs):
        '''
        @ret: [dicts]. Each element is a {key: qkey;
        val: {distribution_num : score}}
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
