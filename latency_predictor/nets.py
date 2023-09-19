import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv
from torch import nn
import pdb

from latency_predictor.transformer import RegressionTransformer
from tst import Transformer
# from latency_predictor.dataset import MAX_LOG_LEN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def torch_avg(data):
    return torch.mean(data["sys_logs"], axis=1).to(device)

class FactorizedLatencyNet(torch.nn.Module):
    def __init__(self,
            cfg,
            num_plan_features,
            num_global_features,
            num_sys_features,
            sys_seq_len,
            layernorm,
            subplan_ests=False,
            out_feats=1,
            final_act="none"):
        super(FactorizedLatencyNet, self).__init__()

        self.fact_arch = cfg["factorized_net"]["arch"]

        if cfg["plan_net"]["arch"] in ["gcn", "gat"]:
            self.gcn_net = SimpleGCN(num_plan_features,
                    num_global_features,
                    cfg["plan_net"]["hl"],
                    cfg["plan_net"]["num_conv_layers"],
                    final_act="none",
                    subplan_ests = cfg["plan_net"]["subplan_ests"],
                    out_feats=cfg["factorized_net"]["embedding_size"],
                    dropout=cfg["plan_net"]["dropout"],
                    arch=cfg["plan_net"]["arch"],
                    )
        elif cfg["plan_net"]["arch"] == "mlp":
            self.job_net = SimpleRegression(num_plan_features,
                cfg["factorized_net"]["embedding_size"],
                cfg["plan_net"]["num_layers"],
                cfg["plan_net"]["hl"])

        if cfg["sys_net"]["arch"] == "mlp":
            self.sys_net = LogAvgRegression(
                    num_sys_features,
                    cfg["factorized_net"]["embedding_size"],
                    cfg["sys_net"]["num_layers"],
                    cfg["sys_net"]["hl"]
                    )
            self.sys_net.to(device)

        elif cfg["sys_net"]["arch"] == "transformer":
            self.sys_net = TransformerLogs(
                    num_sys_features,
                    cfg["factorized_net"]["embedding_size"],
                    cfg["sys_net"]["num_layers"],
                    cfg["sys_net"]["hl"],
                    cfg["sys_net"]["num_heads"],
                    sys_seq_len,
                    layernorm,
                    cfg["sys_net"]["max_pool"],
                    )
            self.sys_net.to(device)

        elif cfg["sys_net"]["arch"] == "avg":
            self.sys_net = torch_avg

        if cfg["factorized_net"]["arch"] == "mlp":

            if cfg["sys_net"]["arch"] == "avg":
                emb_size = cfg["factorized_net"]["embedding_size"] + \
                            num_sys_features
            else:
                emb_size = cfg["factorized_net"]["embedding_size"]*2

            self.fact_net = SimpleRegression(
                    emb_size,
                    1, cfg["factorized_net"]["num_layers"],
                    cfg["factorized_net"]["hl"])
            self.fact_net.to(device)

        elif cfg["factorized_net"]["arch"] == "attention":
            # self.fact_net = RegressionTransformer(
                    # cfg["factorized_net"]["embedding_size"]*2,
                    # 8, 1, 1, 1,
                    # layernorm=True,
                    # ).to(device)
            self.fact_net = AttentionFactNet(
                    cfg["factorized_net"]["embedding_size"]*2,
                    4, cfg["factorized_net"]["num_layers"],
                    1, 1,
                    ).to(device)
            self.fact_net.to(device)

        elif cfg["factorized_net"]["arch"] == "attention2":
            self.fact_net = AttentionFactNet(
                    1,
                    4, 1,
                    cfg["factorized_net"]["embedding_size"]*2,
                    1,
                    ).to(device)
            self.fact_net.to(device)

        elif cfg["factorized_net"]["arch"] == "dot":
            pass

    def forward(self, data):
        xplan = self.gcn_net(data)
        xsys = self.sys_net(data)

        if self.fact_arch == "mlp":
            xplan = xplan.squeeze()
            ## old, w/ batch = 1
            # emb_out = torch.cat([xplan, xsys])
            emb_out = torch.cat([xsys,xplan], axis=-1)
            out = self.fact_net(emb_out)
        elif self.fact_arch == "attention":
            emb_out = torch.cat([xsys,xplan], axis=-1)
            emb_out = emb_out.unsqueeze(dim=1)
            out = self.fact_net(emb_out)

        elif self.fact_arch == "attention2":
            emb_out = emb_out.unsqueeze(dim=2)
            out = self.fact_net(emb_out)

        elif self.fact_arch == "dot":
            out = torch.bmm(xsys.view(xsys.shape[0], 1, xsys.shape[1]),
                    xplan.view(xplan.shape[0], xplan.shape[1], 1)).squeeze()

        return out

class FactorizedLinuxNet(torch.nn.Module):
    def __init__(self,
            cfg,
            num_plan_features,
            num_global_features,
            num_sys_features,
            layernorm,
            subplan_ests=False,
            out_feats=1,
            final_act="none"):
        super(FactorizedLinuxNet, self).__init__()

        self.fact_arch = cfg["factorized_net"]["arch"]
        if cfg["plan_net"]["arch"] == "mlp":
            self.job_net = SimpleRegression(num_plan_features,
                cfg["factorized_net"]["embedding_size"],
                cfg["plan_net"]["num_layers"],
                cfg["plan_net"]["hl"])

        if cfg["sys_net"]["arch"] == "mlp":
            self.sys_net = LogAvgRegression(
                    num_sys_features,
                    cfg["factorized_net"]["embedding_size"],
                    cfg["sys_net"]["num_layers"],
                    cfg["sys_net"]["hl"]
                    )
            self.sys_net.to(device)

        elif cfg["sys_net"]["arch"] == "transformer":
            self.sys_net = TransformerLogs(
                    num_sys_features,
                    cfg["factorized_net"]["embedding_size"],
                    cfg["sys_net"]["num_layers"],
                    cfg["sys_net"]["hl"],
                    cfg["sys_net"]["num_heads"],
                    MAX_LOG_LEN,
                    layernorm,
                    )
            self.sys_net.to(device)

        elif cfg["sys_net"]["arch"] == "avg":
            self.sys_net = torch_avg

        if cfg["factorized_net"]["arch"] == "mlp":

            if cfg["sys_net"]["arch"] == "avg":
                emb_size = cfg["factorized_net"]["embedding_size"] + \
                            num_sys_features
            else:
                emb_size = cfg["factorized_net"]["embedding_size"]*2

            self.fact_net = SimpleRegression(
                    emb_size,
                    1, cfg["factorized_net"]["num_layers"],
                    cfg["factorized_net"]["hl"])
            self.fact_net.to(device)

        elif cfg["factorized_net"]["arch"] == "attention":
            ## TODO: combine this + SimpleRegression module
            # self.fact_net = RegressionTransformer(
                    # cfg["factorized_net"]["embedding_size"]*2,
                    # 4, 1, 1, 1,
                    # layernorm=False,
                    # ).to(device)

            self.fact_net = AttentionFactNet(
                    cfg["factorized_net"]["embedding_size"]*2,
                    4, 1, 1, 1,
                    ).to(device)
            # self.fact_net = AttentionFactNet(
                    # 1,
                    # 4, 1,
                    # cfg["factorized_net"]["embedding_size"]*2,
                    # 1,
                    # ).to(device)

            self.fact_net.to(device)

        elif cfg["factorized_net"]["arch"] == "dot":
            pass

    def forward(self, data):
        # xplan = self.gcn_net(data)
        xplan = self.job_net(data["x"])
        xsys = self.sys_net(data)

        if self.fact_arch == "mlp":
            xplan = xplan.squeeze()
            ## old, w/ batch = 1
            # emb_out = torch.cat([xplan, xsys])
            emb_out = torch.cat([xsys,xplan], axis=-1)
            out = self.fact_net(emb_out)

        elif self.fact_arch == "attention":
            xplan = xplan.squeeze()
            emb_out = torch.cat([xsys,xplan], axis=-1)
            emb_out = emb_out.unsqueeze(dim=1)
            out = self.fact_net(emb_out)

        elif self.fact_arch == "dot":
            out = torch.bmm(xsys.view(xsys.shape[0], 1, xsys.shape[1]),
                    xplan.view(xplan.shape[0], xplan.shape[1], 1)).squeeze()

        return out

class AttentionFactNet(torch.nn.Module):
    def __init__(self, emb, heads, depth, seq_length, num_classes,
            max_pool=True, dropout=0.2, wide=True, layernorm=True):
        super(AttentionFactNet, self).__init__()
        self.trans = RegressionTransformer(
                emb,
                heads, depth, seq_length, emb,
                layernorm=layernorm,
                ).to(device)
        self.final = SimpleRegression(emb,
            num_classes,
            2, 512)

    def forward(self, x):
        #x = data["sys_logs"]
        x = x.to(device, non_blocking=True)
        x = self.trans(x)
        return self.final(x)

        # if len(x.shape) == 2:
            # # batch size is implicitly 1
            # x = x.unsqueeze(dim=0)
        # return self.net(x).squeeze(dim=0)


class TransformerLogs(torch.nn.Module):
    def __init__(self, input_width, n_output,
            num_hidden_layers,
            hidden_layer_size,
            num_heads,
            seq_len,
            layernorm,
            max_pool,
            ):
        super(TransformerLogs, self).__init__()
        # TODO: calculate this
        self.net = RegressionTransformer(input_width,
                num_heads, num_hidden_layers, seq_len, n_output,
                max_pool=max_pool,
                layernorm=layernorm,
                ).to(device)

    def forward(self, data):
        x = data["sys_logs"]
        x = x.to(device, non_blocking=True)

        if len(x.shape) == 2:
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
        print(input_width, n_output)
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
            final_act="none",
            dropout=0.2,
            arch="gcn"
            ):
        super(SimpleGCN, self).__init__()
        assert final_act == "none"

        self.dropout = dropout
        self.final_act=final_act
        self.subplan_ests = subplan_ests
        self.global_hl1 = 32
        self.num_features = num_features
        self.num_global_feats = num_global_feats
        self.out_feats = out_feats
        self.att_heads = 16

        if self.num_global_feats == 0:
            self.global_feats = False
        else:
            self.global_feats = True

        self.hl1 = hl1
        self.num_conv_layers = num_conv_layers

        self.layers = nn.ModuleList()

        if arch == "gcn":
            conv1 = GCNConv(num_features, hl1)
        elif arch == "gat":
            conv1 = GATConv(num_features, hl1,
                    heads = self.att_heads)

        self.layers.append(conv1)

        for i in range(self.num_conv_layers-1):
            # self.layers.append(GCNConv(hl1,hl1))
            if arch == "gcn":
                self.layers.append(GCNConv(hl1,hl1))
            elif arch == "gat":
                self.layers.append(GATConv(self.att_heads*hl1, hl1,
                    heads=self.att_heads))

        if self.subplan_ests:
            self.lin1 = torch.nn.Linear(hl1, hl1)
            self.lin2 = torch.nn.Linear(hl1, hl1)
            self.lin3 = torch.nn.Linear(hl1, self.out_feats)
        else:
            if arch == "gcn":
                self.lin1 = torch.nn.Linear(hl1, hl1)
            elif arch == "gat":
                self.lin1 = torch.nn.Linear(self.att_heads*hl1, hl1)

            if self.global_feats:
                self.global1 = torch.nn.Linear(num_global_feats, self.global_hl1)
                self.lin2 = torch.nn.Linear(hl1+self.global_hl1, hl1)
            else:
                self.lin2 = torch.nn.Linear(hl1, hl1)

            self.lin3 = torch.nn.Linear(hl1, hl1)
            self.lin4 = torch.nn.Linear(hl1, self.out_feats)

        self.do = nn.Dropout(self.dropout)

    def forward(self, data):
        data = data["graph"]
        x, edge_index = data.x, data.edge_index
        x = x.to(device, non_blocking=True)
        x = self.do(x)

        if self.global_feats:
            globalx = data.global_feats

        for layer in self.layers:
            x = F.relu(layer(x, edge_index))

        if self.subplan_ests:
            assert data.num_graphs == 1
            x = self.lin1(x)
            x = F.relu(x)
            x = self.lin2(x)
            x = F.relu(x)
            x = self.lin3(x)
            return x
        else:
            ## code to handle batched training with different shaped graphs
            # print(x.shape)
            # out1 = self.lin4(self.lin3(self.lin2(self.lin1(x))))
            # print(out1.shape)
            # pdb.set_trace()
            prev_idx = 0
            xs = []
            gfs = []
            for i in range(data.num_graphs):
                num_nodes = data[i].x.shape[0]
                ## pooling operator == sum across all nodes;
                xs.append(torch.sum(x[prev_idx:prev_idx+num_nodes], axis=0))
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
            # if self.final_act == "sigmoid":
                # x = torch.sigmoid(x)

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

