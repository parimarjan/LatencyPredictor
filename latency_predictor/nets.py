import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SimpleGCN(torch.nn.Module):
    def __init__(self, num_features, num_global_feats, hl1,
            num_conv_layers,
            subplan_ests=False,
            final_act="none"):
        super(SimpleGCN, self).__init__()

        self.final_act=final_act
        self.subplan_ests = subplan_ests
        self.global_hl1 = 32
        self.num_features = num_features
        self.num_global_feats = num_global_feats
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
            self.lin3 = torch.nn.Linear(hl1, 1)
        else:
            self.lin1 = torch.nn.Linear(hl1, hl1)
            if self.global_feats:
                self.global1 = torch.nn.Linear(num_global_feats, self.global_hl1)
                self.lin2 = torch.nn.Linear(hl1+self.global_hl1, hl1)
            else:
                self.lin2 = torch.nn.Linear(hl1, hl1)
            self.lin3 = torch.nn.Linear(hl1, hl1)
            self.lin4 = torch.nn.Linear(hl1, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

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
            # x = torch.sigmoid(x)
            return x
        else:
            # Alternative, for single graph batch
            # x = torch.sum(x, axis=0)

            # TODO: decide what kind of pooling we want to use;
            # using sum, as it seems to make sense in query cost context?
            # need to do pooling before applying linear layers!
            prev_idx = 0
            # num_glob_feats = data[0].global_feats
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

