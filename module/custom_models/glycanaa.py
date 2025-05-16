from collections.abc import Sequence
import copy

import torch
from torch import nn

from torchdrug import core, layers
from torchdrug.core import Registry as R

from module.custom_data import Glycan


@R.register("models.GlycanAA")
class GlycanAA(nn.Module, core.Configurable):

    def __init__(self, input_dim, hidden_dims, num_unit, num_relation, edge_input_dim=None, short_cut=False,
                 batch_norm=False, activation="relu", concat_hidden=False, readout="sum", readout_type="mono"):
        super(GlycanAA, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        assert input_dim == hidden_dims[0], "Input dimension must equal to hidden dimension."
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1)
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        self.embedding = nn.Embedding(num_unit, input_dim)
        self.atom_layers = nn.ModuleList()
        self.cross_layers = nn.ModuleList()
        self.mono_layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.atom_layers.append(layers.RelationalGraphConv(self.dims[i], self.dims[i + 1], num_relation,
                                                               edge_input_dim, batch_norm, activation))
            self.cross_layers.append(layers.RelationalGraphConv(self.dims[i], self.dims[i + 1], num_relation,
                                                                edge_input_dim, batch_norm, activation))
            self.mono_layers.append(layers.RelationalGraphConv(self.dims[i], self.dims[i + 1], num_relation,
                                                               edge_input_dim, batch_norm, activation))

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        elif readout == "dual":
            self.readout = layers.MeanReadout()
            self.readout_ext = layers.MaxReadout()
            self.output_dim = self.output_dim * 2
        else:
            raise ValueError("Unknown readout `%s`" % readout)
        
        self.readout_type = readout_type
        if readout_type not in {"mono", "all"}:
            raise ValueError(f"Unknown readout type `{readout_type}`")

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

        for idx in range(len(self.atom_layers)):
            hidden1 = self.atom_layers[idx](atom_graph, layer_input)
            
            input2 = torch.zeros_like(layer_input)
            input2[is_atom_node] = hidden1[is_atom_node]
            input2[is_mono_node] = layer_input[is_mono_node]
            hidden2 = self.cross_layers[idx](cross_graph, input2)
            
            input3 = torch.zeros_like(layer_input)
            input3[is_mono_node] = hidden2[is_mono_node]
            hidden3 = self.mono_layers[idx](mono_graph, input3)

            layer_output = torch.zeros_like(layer_input)
            layer_output[is_atom_node] = hidden1[is_atom_node]
            layer_output[is_mono_node] = hidden3[is_mono_node]
            if self.short_cut:
                layer_output = layer_output + layer_input
            hiddens.append(layer_output)
            layer_input = layer_output

        if self.concat_hidden:
            all_node_feature = torch.cat(hiddens, dim=-1)
        else:
            all_node_feature = hiddens[-1]
        
        node_feature = all_node_feature[is_mono_node]
        if self.readout_type == "mono":
            subgraph = graph.node_mask(is_mono_node, compact=True)
            graph_feature = self.readout(subgraph, node_feature)
            if hasattr(self, "readout_ext"):
                graph_feature = torch.cat([graph_feature, self.readout_ext(subgraph, node_feature)], dim=-1)
        elif self.readout_type == "all":
            graph_feature = self.readout(graph, all_node_feature)
            if hasattr(self, "readout_ext"):
                graph_feature = torch.cat([graph_feature, self.readout_ext(graph, all_node_feature)], dim=-1)
        else:
            raise NotImplementedError("Method implemented currently. Location: GlycanAA Readout.")

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature,
            "all_node_feature": all_node_feature,
        }


@R.register("models.GlycanAA_WP")
class GlycanAA_WP(nn.Module, core.Configurable):

    def __init__(self, input_dim, hidden_dims, num_unit, num_relation, edge_input_dim=None, short_cut=False,
                 batch_norm=False, activation="relu", concat_hidden=False, readout="sum"):
        super(GlycanAA_WP, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        assert input_dim == hidden_dims[0], "Input dimension must equal to hidden dimension."
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1)
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        self.embedding = nn.Embedding(num_unit, input_dim)
        self.hidden_layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.hidden_layers.append(layers.RelationalGraphConv(self.dims[i], self.dims[i + 1], num_relation,
                                                               edge_input_dim, batch_norm, activation))

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        elif readout == "dual":
            self.readout = layers.MeanReadout()
            self.readout_ext = layers.MaxReadout()
            self.output_dim = self.output_dim * 2
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, graph, input, all_loss=None, metric=None):
        is_mono_node = graph.unit_type < len(Glycan.units)

        mask = (graph.unit_type == -1)
        input = self.embedding(graph.unit_type.clamp(min=0))
        input[mask] = torch.zeros_like(input[0])
        layer_input = input
        hiddens = []

        for layer in self.hidden_layers:
            hidden = layer(graph, layer_input)
            if self.short_cut:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            all_node_feature = torch.cat(hiddens, dim=-1)
        else:
            all_node_feature = hiddens[-1]
        node_feature = all_node_feature[is_mono_node]
        subgraph = graph.node_mask(is_mono_node, compact=True)
        graph_feature = self.readout(subgraph, node_feature)
        if hasattr(self, "readout_ext"):
            graph_feature = torch.cat([graph_feature, self.readout_ext(subgraph, node_feature)], dim=-1)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature,
            "all_node_feature": all_node_feature,
        }
