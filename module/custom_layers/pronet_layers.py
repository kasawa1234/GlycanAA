import torch
import torch.nn.functional as F
from torch_geometric.nn import inits, MessagePassing
from torch import nn

from torchdrug import layers
from torchdrug.core import Registry as R

"""
Implementation based on https://github.com/divelab/DIG/blob/21476b079c9226f38915dcd082b5c2ee0cddaac8/dig/threedgraph/method/pronet/pronet.py
"""


def swish(x):
    return x * torch.sigmoid(x)

class Linear(nn.Module):
    """
    A linear method encapsulation similar to PyG's.

    Parameters:
        in_channels (int)
        out_channels (int)
        bias (int)
        weight_initializer (string): glorot or zeros
    """

    def __init__(self, in_channels, out_channels, bias=True, weight_initializer='glorot'):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_initializer = weight_initializer

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight_initializer == 'glorot':
            inits.glorot(self.weight)
        elif self.weight_initializer == 'zeros':
            inits.zeros(self.weight)
        if self.bias is not None:
            inits.zeros(self.bias)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class TwoLinear(torch.nn.Module):
    """
    A layer with two linear modules.

    Parameters:
        in_channels (int)
        middle_channels (int)
        out_channels (int)
        bias (bool)
        act (bool)
    """

    def __init__(
            self,
            in_channels,
            middle_channels,
            out_channels,
            bias=False,
            act=False
    ):
        super(TwoLinear, self).__init__()
        self.lin1 = Linear(in_channels, middle_channels, bias=bias)
        self.lin2 = Linear(middle_channels, out_channels, bias=bias)
        self.act = act

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        x = self.lin1(x)
        if self.act:
            x = swish(x)
        x = self.lin2(x)
        if self.act:
            x = swish(x)
        return x


class Aggregate(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j


@R.register("layers.InteractionBlock")
class InteractionBlock(nn.Module):
    def __init__(
            self,
            input_dim,
            all_atom_input_dim,
            hidden_channels,
            output_channels,
            num_layers,
            mid_emb,
            num_relation,
            act=swish,
            dropout=0,
            edge_input_dim=None,
            batch_norm=False,
            activation="relu"
    ):
        super(InteractionBlock, self).__init__()
        self.act = act
        self.dropout = nn.Dropout(dropout)
        
        self.conv0 = layers.RelationalGraphConv(hidden_channels, hidden_channels, num_relation,
                                                edge_input_dim, batch_norm, activation)
        self.conv1 = layers.RelationalGraphConv(hidden_channels, hidden_channels, num_relation,
                                                edge_input_dim, batch_norm, activation)

        self.lin_feature0 = TwoLinear(hidden_channels, mid_emb, hidden_channels)
        self.lin_feature1 = TwoLinear(hidden_channels + all_atom_input_dim, mid_emb, hidden_channels)

        self.lin_1 = Linear(input_dim, hidden_channels)
        self.lin_2 = Linear(input_dim, hidden_channels)

        self.lin0 = Linear(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels)

        self.lins_cat = torch.nn.ModuleList()
        self.lins_cat.append(Linear(2 * hidden_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.lins_cat.append(Linear(hidden_channels, hidden_channels))

        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.final = Linear(hidden_channels, output_channels)

        self.reset_parameters()

    def reset_parameters(self):

        self.lin_feature0.reset_parameters()
        self.lin_feature1.reset_parameters()

        self.lin_1.reset_parameters()
        self.lin_2.reset_parameters()

        self.lin0.reset_parameters()
        self.lin1.reset_parameters()

        for lin in self.lins:
            lin.reset_parameters()
        for lin in self.lins_cat:
            lin.reset_parameters()

        self.final.reset_parameters()


    def forward(self, mono_feature, atom_feature, mono_graph):
        x_lin_1 = self.act(self.lin_1(mono_feature))           # right one
        x_lin_2 = self.act(self.lin_2(mono_feature))           # left one
        
        # the left conv
        input_feature_0 = self.lin_feature0(x_lin_1)
        h0 = self.conv0(mono_graph, input_feature_0)
        h0 = self.lin0(h0)
        h0 = self.act(h0)
        h0 = self.dropout(h0)

        input_feature_1 = self.lin_feature1(torch.cat((x_lin_1, atom_feature), dim=-1))
        h1 = self.conv1(mono_graph, input_feature_1)
        h1 = self.lin1(h1)
        h1 = self.act(h1)
        h1 = self.dropout(h1)

        h = torch.cat((h0, h1),1)
        for lin in self.lins_cat:
            h = self.act(lin(h)) 

        h = h + x_lin_2

        for lin in self.lins:
            h = self.act(lin(h)) 
        h = self.final(h)
        return h