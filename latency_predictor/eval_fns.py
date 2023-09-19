import pandas as pd
import numpy as np
import wandb
import matplotlib.pyplot as plt
from latency_predictor.utils import *

import pdb

def get_eval_fn(loss_name, **kwargs):
    if loss_name == "latency_mse":
        return LatencyMSE()
    elif loss_name == "latency_qerr":
        return LatencyQError()
    elif loss_name == "latency_ae":
        return LatencyAE()
    else:
        print("loss function was: ", loss_name)
        assert False


class LossFunc():
    def __init__(self, **kwargs):
        pass

    def eval(self, ests, trues, **kwargs):
        pass

    def __str__(self):
        return self.__class__.__name__

    # TODO: stuff for saving logs

class LatencyAE():
    def __init__(self, **kwargs):
        pass

    def eval(self, ests, trues, **kwargs):
        ests = np.array(ests)
        trues = np.array(trues)
        # alllosses = np.square(ests - trues)
        alllosses = np.abs(ests-trues)

        return alllosses

    def __str__(self):
        return self.__class__.__name__


class LatencyMSE():
    def __init__(self, **kwargs):
        pass

    def eval(self, ests, trues, **kwargs):
        ests = np.array(ests)
        trues = np.array(trues)
        alllosses = np.square(ests - trues)
        return alllosses

    def __str__(self):
        return self.__class__.__name__


class LatencyQError():

    def __init__(self, **kwargs):
        self.min_est = 1.0

    def eval(self, ests, trues, **kwargs):
        ests = np.array(ests)
        trues = np.array(trues)
        trues = np.maximum(trues, self.min_est)
        ests = np.maximum(ests, self.min_est)

        alllosses = np.maximum((ests / trues), (trues / ests))
        return alllosses

    def __str__(self):
        return self.__class__.__name__
