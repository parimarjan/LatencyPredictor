import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch import nn
import pdb

from latency_predictor.transformer import RegressionTransformer
from tst import Transformer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FactorizedLatencyNet(torch.nn.Module):
    def __init__(self,
            cfg,
            num_plan_features,
            num_global_features,
            num_sys_features,
            subplan_ests=False,
            out_feats=1,
            final_act="none"):
        super(FactorizedLatencyNet, self).__init__()

        self.fact_arch = cfg["factorized_net"]["arch"]
        if cfg["plan_net"]["arch"] == "gcn":
            self.gcn_net = SimpleGCN(num_plan_features,
                    num_global_features,
                    cfg["plan_net"]["hl"],
                    cfg["plan_net"]["num_conv_layers"],
                    final_act="none",
                    subplan_ests = cfg["plan_net"]["subplan_ests"],
                    out_feats=cfg["factorized_net"]["embedding_size"],
                    )

        if cfg["sys_net"]["arch"] == "mlp":
            self.log_net = LogAvgRegression(
                    num_sys_features,
                    cfg["factorized_net"]["embedding_size"],
                    cfg["sys_net"]["num_layers"],
                    cfg["sys_net"]["hl"]
                    )
        elif cfg["sys_net"]["arch"] == "transformer":
            self.log_net = TransformerLogs(
                    num_sys_features,
                    cfg["factorized_net"]["embedding_size"],
                    cfg["sys_net"]["num_layers"],
                    cfg["sys_net"]["hl"]
                    )

        if cfg["factorized_net"]["arch"] == "mlp":
            self.fact_net = SimpleRegression(
                    cfg["factorized_net"]["embedding_size"]*2,
                    1, cfg["factorized_net"]["num_layers"],
                    cfg["factorized_net"]["hl"])

        elif cfg["factorized_net"]["arch"] == "dot":
            pass

    def forward(self, data):

        xplan = self.gcn_net(data)
        xsys = self.log_net(data)

        if self.fact_arch == "mlp":
            xplan = xplan.squeeze()
            emb_out = torch.cat([xplan, xsys])
            out = self.fact_net(emb_out)

        elif self.fact_arch == "dot":
            pass

        return out

class TransformerLogs(torch.nn.Module):
    def __init__(self, input_width, n_output,
            num_hidden_layers,
            hidden_layer_size,
            ):
        super(TransformerLogs, self).__init__()
        # TODO: calculate this
        seq_len = 10
        self.net = RegressionTransformer(input_width,
                4, num_hidden_layers, seq_len, n_output)

    def forward(self, data):
        x = data["sys_logs"]
        # batch size is implicitly 1
        x = x.unsqueeze(dim=0)
        return self.net(x).squeeze(dim=0)

class TSTLogs(torch.nn.Module):
    def __init__(self, input_width, n_output,
            num_hidden_layers,
            hidden_layer_size,
            ):
        super(TSTLogs, self).__init__()
        self.net = Transformer(20, input_width, 2,
                        2, 2, 2, 2)

    def forward(self, data):
        x = data["sys_logs"]
        # if len(x.shape) == 2:
            # x = x.unsqueeze(dim=0)
        return self.net(x)

class LogAvgRegression(torch.nn.Module):
    def __init__(self, input_width, n_output,
            num_hidden_layers,
            hidden_layer_size,
            ):
        super(LogAvgRegression, self).__init__()
        self.net = SimpleRegression(input_width, n_output,
            num_hidden_layers,
            hidden_layer_size)

    def forward(self, data):
        x = data["sys_logs"]
        return self.net(x)

class SimpleRegression(torch.nn.Module):
    def __init__(self, input_width, n_output,
            num_hidden_layers,
            hidden_layer_size,
            final_act="none",
            ):

        super(SimpleRegression, self).__init__()
        self.final_act = final_act
        hidden_layer_size = int(hidden_layer_size)
        self.layers = nn.ModuleList()
        layer1 = nn.Sequential(
            nn.Linear(input_width, hidden_layer_size, bias=True),
            nn.ReLU()
        ).to(device)

        self.layers.append(layer1)

        for i in range(0,num_hidden_layers-1,1):
            layer = nn.Sequential(
                nn.Linear(hidden_layer_size, hidden_layer_size, bias=True),
                nn.ReLU()
            ).to(device)
            self.layers.append(layer)

        final_layer = nn.Sequential(
            nn.Linear(hidden_layer_size, n_output, bias=True),
        ).to(device)
        self.layers.append(final_layer)

    def forward(self, x):
        x = x.to(device, non_blocking=True)
        output = x
        for layer in self.layers:
            output = layer(output)

        if self.final_act == "sigmoid":
            output = torch.sigmoid(output)
        return output

class SimpleGCN(torch.nn.Module):
    def __init__(self,
            num_features, num_global_feats, hl1,
            num_conv_layers,
            subplan_ests=False,
            out_feats=1,
            final_act="none"):
        super(SimpleGCN, self).__init__()

        self.final_act=final_act
        self.subplan_ests = subplan_ests
        self.global_hl1 = 32
        self.num_features = num_features
        self.num_global_feats = num_global_feats
        self.out_feats = out_feats

        if self.num_global_feats == 0:
            self.global_feats = False
        else:
            self.global_feats = True

        self.hl1 = hl1
        self.num_conv_layers = num_conv_layers

        self.conv1 = GCNConv(num_features, hl1)

        if self.num_conv_layers == 2:
            self.conv2 = GCNConv(hl1, hl1)
        elif self.num_conv_layers == 4:
            self.conv2 = GCNConv(hl1, hl1)
            self.conv3 = GCNConv(hl1, hl1)
            self.conv4 = GCNConv(hl1, hl1)

        else:
            assert False

        if self.subplan_ests:
            self.lin1 = torch.nn.Linear(hl1, hl1)
            self.lin2 = torch.nn.Linear(hl1, hl1)
            self.lin3 = torch.nn.Linear(hl1, self.out_feats)
        else:
            self.lin1 = torch.nn.Linear(hl1, hl1)
            if self.global_feats:
                self.global1 = torch.nn.Linear(num_global_feats, self.global_hl1)
                self.lin2 = torch.nn.Linear(hl1+self.global_hl1, hl1)
            else:
                self.lin2 = torch.nn.Linear(hl1, hl1)
            self.lin3 = torch.nn.Linear(hl1, hl1)
            self.lin4 = torch.nn.Linear(hl1, self.out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.to(device, non_blocking=True)

        if self.global_feats:
            globalx = data.global_feats

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        if self.num_conv_layers == 2:
            x = self.conv2(x, edge_index)
            x = F.relu(x)
        elif self.num_conv_layers == 4:
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = self.conv3(x, edge_index)
            x = F.relu(x)
            x = self.conv4(x, edge_index)
            x = F.relu(x)

        if self.subplan_ests:
            assert data.num_graphs == 1
            x = self.lin1(x)
            x = F.relu(x)
            x = self.lin2(x)
            x = F.relu(x)
            x = self.lin3(x)
            return x
        else:
            # TODO: decide what kind of pooling we want to use;
            # using sum, as it seems to make sense in query cost context?
            # need to do pooling before applying linear layers!
            prev_idx = 0
            xs = []
            gfs = []
            for i in range(data.num_graphs):
                num_nodes = data[i].x.shape[0]
                xs.append(torch.sum(x[prev_idx:num_nodes], axis=0))
                if self.global_feats:
                    sidx = i*self.num_global_feats
                    gfs.append(globalx[sidx:sidx+self.num_global_feats])
                prev_idx += num_nodes

            x = torch.stack(xs)
            if self.global_feats:
                globalx = torch.stack(gfs)
                globalx = self.global1(globalx)
                globalx = F.relu(globalx)

            # fully connected part
            x = self.lin1(x)
            x = F.relu(x)

            # concat them
            if self.global_feats:
                x = torch.cat([x, globalx], axis=1)

            x = self.lin2(x)
            x = F.relu(x)
            x = self.lin3(x)
            x = F.relu(x)
            x = self.lin4(x)
            if self.final_act == "sigmoid":
                x = torch.sigmoid(x)

            return x.squeeze(1)

# from TreeConvolution.util import prepare_trees
# from TreeConvolution.tcnn import *
# from latency_predictor.dataset import *

# class TCNN2(torch.nn.Module):
    # def __init__(self, num_features, num_global_feats, hl1,
            # final_act="sigmoid"):
        # super(TCNN2, self).__init__()
        # self.final_act = final_act
        # self.cuda = "cuda" in str(device)
        # self.global_hl1 = 32
        # self.hl1 = hl1
        # self.hl2 = int(self.hl1 / 2)
        # self.hl3 = int(self.hl2 / 2)
        # self.hl4 = int(self.hl3 / 2)

        # self.tcnn = torch.nn.Sequential(
            # BinaryTreeConv(num_features, self.hl1),
            # TreeLayerNorm(),
            # TreeActivation(torch.nn.LeakyReLU()),
            # BinaryTreeConv(self.hl1, self.hl2),
            # TreeLayerNorm(),
            # TreeActivation(torch.nn.LeakyReLU()),
            # BinaryTreeConv(self.hl2, self.hl3),
            # TreeLayerNorm(),
            # DynamicPooling())

        # self.lin1 = torch.nn.Linear(self.hl3, self.hl4)
        # self.global1 = torch.nn.Linear(num_global_feats, self.global_hl1)
        # comb_layer_size = self.hl4 + self.global_hl1
        # self.lin2 = torch.nn.Linear(comb_layer_size, comb_layer_size)
        # self.lin3 = torch.nn.Linear(comb_layer_size, 1)

    # def forward(self, data):
        # x = data["X"]
        # globalx = data["global_feats"]
        # if not isinstance(x[0], torch.Tensor):
            # x = prepare_trees(x, transformer, left_child, right_child,
                    # self.cuda)
        # else:
            # assert False

        # x = self.tcnn(x)
        # x = self.lin1(x)
        # x = F.relu(x)
        # globalx = self.global1(globalx)
        # globalx = F.relu(globalx)

        # # concat them
        # x = torch.cat([x, globalx], axis=-1)
        # x = self.lin2(x)
        # x = F.relu(x)
        # x = self.lin3(x)

        # if self.final_act == "sigmoid":
            # return torch.sigmoid(x)
        # else:
            # return x

