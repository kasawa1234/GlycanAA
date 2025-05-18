from collections.abc import Sequence

import os
import sys
import copy
import torch
from torch import nn
from torch_scatter import scatter_add

from torchdrug import core, layers, models
from torchdrug.core import Registry as R

from module.custom_layers import CompGCNConv, SimpleSpatialLineGraph
from module.custom_data import Glycan

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from module.custom_layers import SAMBlock, InteractionBlock, Aggregate


@R.register("models.GlycanGCN")
class GlycanGCN(models.GCN):
    """
    Graph Convolutional Network proposed in `Semi-Supervised Classification with Graph Convolutional Networks`_.

    .. _Semi-Supervised Classification with Graph Convolutional Networks:
        https://arxiv.org/pdf/1609.02907.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        num_unit (int): number of monosaccharide units
        edge_input_dim (int, optional): dimension of edge features
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum``, ``mean`` and ``dual``.
    """

    def __init__(self, input_dim, hidden_dims, num_unit, edge_input_dim=None, short_cut=False, batch_norm=False,
                 activation="relu", concat_hidden=False, readout="sum"):
        super(GlycanGCN, self).__init__(input_dim, hidden_dims, edge_input_dim, short_cut, batch_norm,
                                        activation, concat_hidden, readout.replace("dual", "mean"))

        self.embedding = nn.Embedding(num_unit, input_dim)
        if readout == "dual":
            self.readout_ext = layers.MaxReadout()
            self.node_output_dim = self.output_dim
            self.output_dim = self.output_dim * 2

    def forward(self, graph, input, all_loss=None, metric=None):
        input = self.embedding(graph.unit_type)
        feature = super(GlycanGCN, self).forward(graph, input, all_loss, metric)

        if hasattr(self, "readout_ext"):
            node_feature, graph_feature = feature["node_feature"], feature["graph_feature"]
            feature["graph_feature"] = torch.cat([graph_feature, self.readout_ext(graph, node_feature)], dim=-1)

        return feature


@R.register("models.GlycanRGCN")
class GlycanRGCN(models.RGCN):
    """
    Relational Graph Convolutional Network proposed in `Modeling Relational Data with Graph Convolutional Networks?`_.

    .. _Modeling Relational Data with Graph Convolutional Networks?:
        https://arxiv.org/pdf/1703.06103.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        num_unit (int): number of monosaccharide units
        num_relation (int): number of relations
        edge_input_dim (int, optional): dimension of edge features
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum``, ``mean`` and ``dual``.
    """

    def __init__(self, input_dim, hidden_dims, num_unit, num_relation, edge_input_dim=None, short_cut=False,
                 batch_norm=False,
                 activation="relu", concat_hidden=False, readout="sum"):
        super(GlycanRGCN, self).__init__(input_dim, hidden_dims, num_relation, edge_input_dim, short_cut, batch_norm,
                                         activation, concat_hidden, readout.replace("dual", "mean"))

        self.embedding = nn.Embedding(num_unit, input_dim)
        if readout == "dual":
            self.readout_ext = layers.MaxReadout()
            self.node_output_dim = self.output_dim
            self.output_dim = self.output_dim * 2

    def forward(self, graph, input, all_loss=None, metric=None):
        input = self.embedding(graph.unit_type)
        feature = super(GlycanRGCN, self).forward(graph, input, all_loss, metric)

        if hasattr(self, "readout_ext"):
            node_feature, graph_feature = feature["node_feature"], feature["graph_feature"]
            feature["graph_feature"] = torch.cat([graph_feature, self.readout_ext(graph, node_feature)], dim=-1)

        return feature


@R.register("models.GlycanGAT")
class GlycanGAT(models.GAT):
    """
    Graph Attention Network proposed in `Graph Attention Networks`_.

    .. _Graph Attention Networks:
        https://arxiv.org/pdf/1710.10903.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        num_unit (int): number of monosaccharide units
        edge_input_dim (int, optional): dimension of edge features
        num_head (int, optional): number of attention heads
        negative_slope (float, optional): negative slope of leaky relu activation
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum``, ``mean`` and ``dual``.
    """

    def __init__(self, input_dim, hidden_dims, num_unit, edge_input_dim=None, num_head=1, negative_slope=0.2,
                 short_cut=False,
                 batch_norm=False, activation="relu", concat_hidden=False, readout="sum"):
        super(GlycanGAT, self).__init__(input_dim, hidden_dims, edge_input_dim, num_head, negative_slope, short_cut,
                                        batch_norm,
                                        activation, concat_hidden, readout.replace("dual", "mean"))

        self.embedding = nn.Embedding(num_unit, input_dim)
        if readout == "dual":
            self.readout_ext = layers.MaxReadout()
            self.node_output_dim = self.output_dim
            self.output_dim = self.output_dim * 2

    def forward(self, graph, input, all_loss=None, metric=None):
        input = self.embedding(graph.unit_type)
        feature = super(GlycanGAT, self).forward(graph, input, all_loss, metric)

        if hasattr(self, "readout_ext"):
            node_feature, graph_feature = feature["node_feature"], feature["graph_feature"]
            feature["graph_feature"] = torch.cat([graph_feature, self.readout_ext(graph, node_feature)], dim=-1)

        return feature


@R.register("models.GlycanGIN")
class GlycanGIN(models.GIN):
    """
    Graph Ismorphism Network proposed in `How Powerful are Graph Neural Networks?`_

    .. _How Powerful are Graph Neural Networks?:
        https://arxiv.org/pdf/1810.00826.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        num_unit (int): number of monosaccharide units
        edge_input_dim (int, optional): dimension of edge features
        num_mlp_layer (int, optional): number of MLP layers
        eps (int, optional): initial epsilon
        learn_eps (bool, optional): learn epsilon or not
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum``, ``mean`` and ``dual``.
    """

    def __init__(self, input_dim, hidden_dims, num_unit, edge_input_dim=None, num_mlp_layer=2, eps=0, learn_eps=False,
                 short_cut=False, batch_norm=False, activation="relu", concat_hidden=False, readout="sum"):
        super(GlycanGIN, self).__init__(input_dim, hidden_dims, edge_input_dim, num_mlp_layer, eps, learn_eps,
                                        short_cut, batch_norm, activation, concat_hidden,
                                        readout.replace("dual", "mean"))

        self.embedding = nn.Embedding(num_unit, input_dim)
        if readout == "dual":
            self.readout_ext = layers.MaxReadout()
            self.node_output_dim = self.output_dim
            self.output_dim = self.output_dim * 2

    def forward(self, graph, input, all_loss=None, metric=None):
        input = self.embedding(graph.unit_type)
        feature = super(GlycanGIN, self).forward(graph, input, all_loss, metric)

        if hasattr(self, "readout_ext"):
            node_feature, graph_feature = feature["node_feature"], feature["graph_feature"]
            feature["graph_feature"] = torch.cat([graph_feature, self.readout_ext(graph, node_feature)], dim=-1)

        return feature


@R.register("models.GlycanCompGCN")
class GlycanCompGCN(nn.Module, core.Configurable):

    def __init__(self, input_dim, hidden_dims, num_relation, num_unit, edge_input_dim=None, short_cut=False,
                 batch_norm=False, activation="relu", concat_hidden=False, readout="sum", composition="multiply"):
        super(GlycanCompGCN, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1)
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        self.embedding_init = nn.Embedding(num_unit, input_dim)
        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(CompGCNConv(self.dims[i], self.dims[i + 1], num_relation, edge_input_dim,
                                           batch_norm, activation, composition))

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        elif readout == "max":
            self.readout = layers.MaxReadout()
        elif readout == "dual":
            self.readout1, self.readout2 = layers.MeanReadout(), layers.MaxReadout()
            self.node_output_dim = self.output_dim
            self.output_dim = self.output_dim * 2
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, graph, input, all_loss=None, metric=None):
        hiddens = []
        layer_input = self.embedding_init(graph.unit_type)

        for layer in self.layers:
            hidden = layer(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]

        if hasattr(self, "readout1"):
            graph_feature = torch.cat([self.readout1(graph, node_feature), self.readout2(graph, node_feature)], dim=-1)
        else:
            graph_feature = self.readout(graph, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }


@R.register("models.GlycanGearNet")
class GlycanGearNet(nn.Module, core.Configurable):
    """
    Geometry Aware Relational Graph Neural Network proposed in
    `Protein Representation Learning by Geometric Structure Pretraining`_.

    .. _Protein Representation Learning by Geometric Structure Pretraining:
        https://arxiv.org/pdf/2203.06125.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        num_relation (int): number of relations
        edge_input_dim (int, optional): dimension of edge features
        num_angle_bin (int, optional): number of bins to discretize angles between edges.
            The discretized angles are used as relations in edge message passing.
            If not provided, edge message passing is disabled.
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum`` and ``mean``.
    """
    def __init__(self, input_dim, hidden_dims, num_unit, num_relation, edge_input_dim=None, num_angle_bin=None,
                 short_cut=False, batch_norm=False, activation="relu", concat_hidden=False, readout="sum"):
        super(GlycanGearNet, self).__init__()

        self.embedding = nn.Embedding(num_unit, input_dim)

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.dims = [input_dim] + list(hidden_dims)
        self.edge_dims = [edge_input_dim] + self.dims[:-1]
        self.num_relation = num_relation
        self.num_angle_bin = num_angle_bin
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.batch_norm = batch_norm

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layers.GeometricRelationalGraphConv(self.dims[i], self.dims[i + 1], num_relation,
                                                                   None, batch_norm, activation))
        if num_angle_bin:
            self.spatial_line_graph = SimpleSpatialLineGraph(num_angle_bin)
            self.edge_layers = nn.ModuleList()
            for i in range(len(self.edge_dims) - 1):
                self.edge_layers.append(layers.GeometricRelationalGraphConv(
                    self.edge_dims[i], self.edge_dims[i + 1], num_angle_bin, None, batch_norm, activation))

        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 1):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        elif readout == "dual":
            self.readout_ext = layers.MaxReadout()
            self.node_output_dim = self.output_dim
            self.output_dim = self.output_dim * 2
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Compute the node representations and the graph representation(s).

        Parameters:
            graph (Graph): :math:`n` graph(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``node_feature`` and ``graph_feature`` fields:
                node representations of shape :math:`(|V|, d)`, graph representations of shape :math:`(n, d)`
        """
        input = self.embedding(graph.unit_type)
        hiddens = []
        layer_input = input
        if self.num_angle_bin:
            line_graph = self.spatial_line_graph(graph)
            edge_input = line_graph.node_feature.float()

        for i in range(len(self.layers)):
            hidden = self.layers[i](graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            if self.num_angle_bin:
                edge_hidden = self.edge_layers[i](line_graph, edge_input)
                edge_weight = graph.edge_weight.unsqueeze(-1)
                node_out = graph.edge_list[:, 1] * self.num_relation + graph.edge_list[:, 2]
                update = scatter_add(edge_hidden * edge_weight, node_out, dim=0,
                                     dim_size=graph.num_node * self.num_relation)
                update = update.view(graph.num_node, self.num_relation * edge_hidden.shape[1])
                update = self.layers[i].linear(update)
                update = self.layers[i].activation(update)
                hidden = hidden + update
                edge_input = edge_hidden
            if self.batch_norm:
                hidden = self.batch_norms[i](hidden)
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        graph_feature = self.readout(graph, node_feature)

        if hasattr(self, "readout_ext"):
            graph_feature = torch.cat([graph_feature, self.readout_ext(graph, node_feature)], dim=-1)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }

@R.register("models.VabsNet")
class VabsNet(nn.Module, core.Configurable):
    def __init__(self, num_unit, hidden_dim, edge_dim, short_cut=False, concat_hidden=False,
                 num_layers=12, num_heads=12, dropout=0.1, readout="sum"):
        super(VabsNet, self).__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = 2 * hidden_dim * (num_layers if concat_hidden else 1)
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        self.sam_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.sam_layers.append(SAMBlock(hidden_dim, hidden_dim, edge_dim, num_heads, dropout))

        self.embedding_mono = nn.Embedding(num_unit, hidden_dim)
        self.embedding_atom = nn.Embedding(num_unit, hidden_dim)

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        else:
            raise ValueError("Unknown readout `%s`" % readout)
        
    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Compute the node representations and the graph representation(s).

        Require the graph(s) to have the same number of relations as this module.

        Parameters:
            graph (Graph): :math:`n` graph(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``node_feature`` and ``graph_feature`` fields:
                node representations of shape :math:`(|V|, d)`, graph representations of shape :math:`(n, d)`
        """
        is_atom_node = graph.unit_type >= len(Glycan.units)
        is_mono_node = ~is_atom_node
        node_in, node_out, _ = graph.edge_list.t()
        node_in_type = graph.unit_type[node_in]
        node_out_type = graph.unit_type[node_out]

        atom_graph = graph

        mono_graph = copy.deepcopy(graph)
        mono_mask = (node_in_type < len(Glycan.units)) & (node_out_type < len(Glycan.units))
        mono_graph._edge_list = graph.edge_list[mono_mask]
        mono_graph._edge_weight = graph.edge_weight[mono_mask]
        mono_graph.edge_feature = graph.edge_feature[mono_mask]

        mask = (graph.unit_type == -1)
        mono_input = self.embedding_mono(graph.unit_type.clamp(min=0))
        atom_input = self.embedding_atom(graph.unit_type.clamp(min=0))
        mono_input[mask] = torch.zeros_like(mono_input[0])
        atom_input[mask] = torch.zeros_like(atom_input[0])

        mono_layer_input = mono_input
        atom_layer_input = atom_input
        hiddens = []

        for layer in self.sam_layers:
            atom_hidden, mono_hidden = layer(atom_graph, atom_layer_input, mono_graph, mono_layer_input)

            if self.short_cut:
                atom_hidden = atom_hidden + atom_layer_input
                mono_hidden = mono_hidden + mono_layer_input
            
            hiddens.append(torch.cat([atom_hidden, mono_hidden], dim=-1))
            
            atom_layer_input = atom_hidden
            mono_layer_input = mono_hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        graph_feature = self.readout(graph, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }


@R.register("models.ProNet")
class ProNet(nn.Module, core.Configurable):
    """
    Graph Convolutional Network proposed in `Semi-Supervised Classification with Graph Convolutional Networks`_.

    .. _Learning Hierarchical Protein Representations via Complete 3D Graph Networks:
        https://arxiv.org/pdf/2207.12600

    Implementation of ProNet bases on https://github.com/divelab/DIG/blob/21476b079c9226f38915dcd082b5c2ee0cddaac8/dig/threedgraph/method/pronet/pronet.py
    """

    def __init__(self, hidden_dim, num_blocks, num_unit, num_relation, edge_input_dim=None, short_cut=False,
                 batch_norm=False, activation="relu", concat_hidden=False, readout="sum"):
        super(ProNet, self).__init__()

        self.input_dim = hidden_dim
        self.output_dim = hidden_dim * (num_blocks if concat_hidden else 1)
        self.dims = [hidden_dim] * (num_blocks + 1)
        self.num_relation = num_relation
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        self.embedding = nn.Embedding(num_unit, hidden_dim)
        self.interaction_blocks = nn.ModuleList(
            [
                InteractionBlock(
                    input_dim=hidden_dim,
                    all_atom_input_dim=hidden_dim,
                    hidden_channels=hidden_dim,
                    output_channels=hidden_dim,
                    num_layers=3,
                    mid_emb=128,
                    num_relation=num_relation,
                    dropout=0,
                    edge_input_dim=edge_input_dim,
                    batch_norm=batch_norm,
                    activation=activation
                )
                for _ in range(num_blocks)
            ]
        )

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        elif readout == "dual":
            self.readout = layers.MeanReadout()
            self.readout_ext = layers.MaxReadout()
            self.node_output_dim = self.output_dim
            self.output_dim = self.output_dim * 2
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, graph, input, all_loss=None, metric=None):
        is_atom_node = graph.unit_type >= len(Glycan.units)
        is_mono_node = ~is_atom_node
        node_in, node_out, _ = graph.edge_list.t()
        node_in_type = graph.unit_type[node_in]
        node_out_type = graph.unit_type[node_out]

        atom_graph = copy.deepcopy(graph)
        atom_mask = (node_in_type >= len(Glycan.units)) & (node_out_type >= len(Glycan.units))
        atom_graph._edge_list = graph.edge_list[atom_mask]
        atom_graph._edge_weight = graph.edge_weight[atom_mask]
        atom_graph.edge_feature = graph.edge_feature[atom_mask]

        cross_graph = copy.deepcopy(graph)
        cross_mask = (node_in_type >= len(Glycan.units)) & (node_out_type < len(Glycan.units))
        cross_graph._edge_list = graph.edge_list[cross_mask]
        cross_graph._edge_weight = graph.edge_weight[cross_mask]
        cross_graph.edge_feature = graph.edge_feature[cross_mask]

        mono_graph = copy.deepcopy(graph)
        mono_mask = (node_in_type < len(Glycan.units)) & (node_out_type < len(Glycan.units))
        mono_graph._edge_list = graph.edge_list[mono_mask]
        mono_graph._edge_weight = graph.edge_weight[mono_mask]
        mono_graph.edge_feature = graph.edge_feature[mono_mask]

        mask = (graph.unit_type == -1)
        input = self.embedding(graph.unit_type.clamp(min=0))
        input[mask] = torch.zeros_like(input[0])
        layer_input = input
        hiddens = []
            
        atom_feature_all = input
        edge_index = cross_graph._edge_list[:, :2].t()
        sum_module = Aggregate()
        atom_feature = sum_module(atom_feature_all, edge_index)

        for idx in range(len(self.interaction_blocks)):
            layer_output = self.interaction_blocks[idx](layer_input, atom_feature, mono_graph)
            if self.short_cut:
                layer_output = layer_output + layer_input
            hiddens.append(layer_output)
            layer_input = layer_output

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        node_feature = node_feature[is_mono_node]
        subgraph = graph.node_mask(is_mono_node, compact=True)
        graph_feature = self.readout(subgraph, node_feature)
        if hasattr(self, "readout_ext"):
            graph_feature = torch.cat([graph_feature, self.readout_ext(subgraph, node_feature)], dim=-1)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }
