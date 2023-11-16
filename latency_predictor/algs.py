import networkx as nx
# import xgboost as xgb
import wandb
import random
import numpy as np
from collections import defaultdict
import pdb
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.metrics import mean_squared_error,mean_absolute_error

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

class DBMS(LatencyPredictor):
    def __init__(self, *args, **kwargs):
        # TODO: set each of the kwargs as variables
        for k, val in kwargs.items():
            self.__setattr__(k, val)

    def __str__(self):
        return "DBMS-" + self.granularity

    def train(self, train_plans, sys_logs, featurizer,
            **kwargs):

        self.featurizer = featurizer
        self.linear_models = {}
        costs = defaultdict(list)
        latencies = defaultdict(list)
        tags = set()

        for plan in train_plans:
            if plan.graph["concurrent"] and self.only_single:
                continue

            tags.add(plan.graph["tag"])

            latency = plan.graph["latency"]
            instance = plan.graph["lt_type"]
            nc = plan.graph["num_concurrent"]

            pdata = dict(plan.nodes(data=True))
            max_cost = max(subdict['TotalCost'] for subdict in
                    pdata.values())

            if self.normy == "log":
                latency = np.log(latency)
                max_cost = np.log(max_cost)

            if self.granularity == "all":
                costs["all"].append(max_cost)
                latencies["all"].append(latency)
            elif self.granularity == "template":
                key = plan.graph["lt_type"] + str(plan.graph["template"])
                costs[key].append(max_cost)
                latencies[key].append(latency)

            elif self.granularity == "mpl":
                key = instance + str(nc)
                costs[key].append(max_cost)
                latencies[key].append(latency)
            else:
                costs[instance].append(max_cost)
                latencies[instance].append(latency)

        for lt,cur_costs in costs.items():
            # Reshape X to be a 2D array (necessary for scikit-learn's `fit`
            # method)
            cur_lats = latencies[lt]
            X_reshaped = np.array(cur_costs).reshape(-1, 1)
            Y = np.array(cur_lats)
            # Initialize the model
            model = LinearRegression()
            # Fit the model
            model.fit(X_reshaped, Y)

            # Predict
            predictions = model.predict(X_reshaped)
            # Calculate Mean Squared Error
            mse = mean_squared_error(Y, predictions)
            print(f"Mean Squared Error: {mse:.2f}")

            ae = mean_absolute_error(Y, predictions)
            print(f"Abs Squared Error: {ae:.2f}")

            # Print out the coefficients
            print("Slope (Coefficient for X):", model.coef_[0])
            print("Intercept:", model.intercept_)

            self.linear_models[lt] = model

        # Assuming the class method context where `self.linear_models` exists:
        # with open("baselines/linear_models.pkl", "wb") as file:
            # pickle.dump(self.linear_models, file)

    def test(self, plans, sys_logs, **kwargs):
        '''
        '''
        # if self.fit_test and "unseen" not in kwargs["samples_type"]:
            # self.train(plans, sys_logs, self.featurizer)

        ret = []
        for plan in plans:
            if self.granularity == "all":
                lt = "all"
            elif self.granularity == "mpl":
                lt = plan.graph["lt_type"] + str(plan.graph["num_concurrent"])
            elif self.granularity == "template":
                lt = plan.graph["lt_type"] + str(plan.graph["template"])
            else:
                lt = plan.graph["lt_type"]

            pdata = dict(plan.nodes(data=True))
            max_cost = max(subdict['TotalCost'] for subdict in
                    pdata.values())
            if self.normy == "log":
                max_cost = np.log(max_cost)

            if lt not in self.linear_models:
                print(self.linear_models.keys())
                print(lt)
                pdb.set_trace()
            model = self.linear_models[lt]
            X_reshaped = np.array([max_cost]).reshape(-1, 1)
            prediction = model.predict(X_reshaped)
            if self.normy == "log":
                ret.append(np.exp(prediction[0]))
            else:
                ret.append(prediction[0])

        return ret
