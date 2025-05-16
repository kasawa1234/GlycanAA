import copy
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_min

from torchdrug import core, tasks, layers
from torchdrug.data import constant
from torchdrug.layers import functional
from torchdrug.core import Registry as R

from module.custom_data.glycan import Glycan, PackedGlycan


@R.register("tasks.HeteroGlycanAttributeMasking")
class HeteroGlycanAttributeMasking(tasks.Task, core.Configurable):

    mono_mask_token = Glycan.units.index("Unknown")
    atom_mask_token = constant.ATOM_SYMBOL.index("Null") + len(Glycan.units)

    def __init__(self, model, mono_mask_rate=0.15, atom_mask_rate=0.15, num_mlp_layer=2):
        super(HeteroGlycanAttributeMasking, self).__init__()
        self.model = model
        self.mono_mask_rate = mono_mask_rate
        self.atom_mask_rate = atom_mask_rate
        self.num_mlp_layer = num_mlp_layer

    def preprocess(self, train_set, valid_set, test_set):
        if hasattr(self.model, "node_output_dim"):
            model_output_dim = self.model.node_output_dim
        else:
            model_output_dim = self.model.output_dim
        num_label = 210
        self.mlp = layers.MLP(model_output_dim, [model_output_dim] * (self.num_mlp_layer - 1) + [num_label])

    def predict_and_target(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        cumsums = graph.cumsums
        num_nodes = graph.num_nodes
        num_monos = torch.tensor([x[0].cpu() for x in cumsums], dtype=torch.long, device=self.device)
        num_atoms = num_nodes - num_monos
        num_cum_nodes = num_nodes.cumsum(0)
        num_cum_monos = num_monos.cumsum(0)
        max_index = graph.unit_type.size(0) - 1

        if self.mono_mask_rate > 0:
            num_samples_mono = (num_monos * self.mono_mask_rate).long().clamp(1)
            num_sample_mono = num_samples_mono.sum()
            mono_sample2graph = torch.repeat_interleave(num_samples_mono)

            node_subindex_mono = (torch.rand(num_sample_mono, device=self.device) * num_monos[mono_sample2graph]).long()
            node_index_mono = node_subindex_mono + (num_cum_monos - num_monos)[mono_sample2graph]
            all_node_index_mono = node_subindex_mono + (num_cum_nodes - num_nodes)[mono_sample2graph]

            node_subindex_atom = []
            num_samples_atom = torch.zeros_like(num_samples_mono)
            for i, index in enumerate(node_subindex_mono):
                graph_id = mono_sample2graph[i]

                node_subindex_atom.extend([x for x in range(cumsums[graph_id][index], cumsums[graph_id][index + 1])])
                num_samples_atom[graph_id] += cumsums[graph_id][index + 1] - cumsums[graph_id][index]

            node_subindex_atom = torch.tensor(node_subindex_atom, dtype=torch.long, device=self.device)
            sample2graph_atom = torch.repeat_interleave(num_samples_atom)
            node_index_atom = node_subindex_atom + (num_cum_nodes - num_nodes)[sample2graph_atom]

            all_node_index_mono = all_node_index_mono.clamp(min=0, max=max_index)
            node_index_atom = node_index_atom.clamp(min=0, max=max_index)
        else:
            all_node_index_mono = torch.zeros((0,), dtype=torch.long, device=self.device)
            node_index_atom = torch.zeros((0,), dtype=torch.long, device=self.device)
            node_index_mono = torch.zeros((0,), dtype=torch.long, device=self.device)

        if self.atom_mask_rate > 0:
            num_samples_atom = (num_atoms * self.atom_mask_rate).long()
            num_sample_atom = num_samples_atom.sum()
            atom_sample2graph = torch.repeat_interleave(num_samples_atom)

            atom_subindex = (torch.rand(num_sample_atom, device=self.device) * num_atoms[atom_sample2graph]).long()
            atom_index = atom_subindex + (num_cum_nodes - num_nodes)[atom_sample2graph] + num_monos[atom_sample2graph]
            atom_index = atom_index.clamp(min=0, max=max_index)
        else:
            atom_index = torch.zeros((0,), dtype=torch.long, device=self.device)

        all_node_index_mono = torch.unique(all_node_index_mono)
        all_node_index_atom = torch.unique(torch.cat([node_index_atom, atom_index], dim=0))
        mono_target = graph.unit_type[all_node_index_mono]
        atom_target = graph.unit_type[all_node_index_atom]
        input = graph.node_feature.float()
        graph.unit_type[all_node_index_mono] = self.mono_mask_token
        graph.unit_type[all_node_index_atom] = self.atom_mask_token

        output = self.model(graph, input, all_loss, metric)
        mono_feature = output["all_node_feature"][all_node_index_mono]
        mono_pred = self.mlp(mono_feature)
        atom_feature = output["all_node_feature"][all_node_index_atom]
        atom_pred = self.mlp(atom_feature)

        return mono_pred, atom_pred, mono_target, atom_target

    def evaluate(self, mono_pred, atom_pred, mono_target, atom_target):
        metric = {}
        mono_acc = (mono_pred.argmax(dim=-1) == mono_target).float().mean()
        atom_acc = (atom_pred.argmax(dim=-1) == atom_target).float().mean()

        name = tasks._get_metric_name("acc")
        metric["%s [monosaccharide]" % name] = mono_acc
        metric["%s [atom]" % name] = atom_acc

        return metric

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        mono_pred, atom_pred, mono_target, atom_target = self.predict_and_target(batch, all_loss, metric)
        metric.update(self.evaluate(mono_pred, atom_pred, mono_target, atom_target))

        mono_loss = F.cross_entropy(mono_pred, mono_target)
        atom_loss = F.cross_entropy(atom_pred, atom_target)
        name = tasks._get_criterion_name("ce")
        metric["%s [monosaccharide]" % name] = mono_loss
        metric["%s [atom]" % name] = atom_loss

        all_loss += mono_loss
        all_loss += atom_loss

        return all_loss, metric


@R.register("tasks.GlycanAttributeMasking")
class GlycanAttributeMasking(tasks.Task, core.Configurable):
    """
    Attribute masking proposed in `Strategies for Pre-training Graph Neural Networks`_.

    .. _Strategies for Pre-training Graph Neural Networks:
        https://arxiv.org/abs/1905.12265

    Parameters:
    - model (nn.Module): node representation model
    - mask_rate (float, optional): rate of masked nodes
    - num_mlp_layer (int, optional): number of MLP layers
    """

    def __init__(self, model, mask_rate=0.15, num_mlp_layer=2):
        super(GlycanAttributeMasking, self).__init__()
        self.model = model
        self.mask_rate = mask_rate
        self.num_mlp_layer = num_mlp_layer

    def preprocess(self, train_set, valid_set, test_set):
        if hasattr(self.model, "node_output_dim"):
            model_output_dim = self.model.node_output_dim
        else:
            model_output_dim = self.model.output_dim
        num_label = 143
        self.mlp = layers.MLP(model_output_dim, [model_output_dim] * (self.num_mlp_layer - 1) + [num_label])

    def predict_and_target(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]

        num_nodes = graph.num_nodes
        num_cum_nodes = num_nodes.cumsum(0)
        num_samples = (num_nodes * self.mask_rate).long().clamp(1)

        sample2graph = torch.repeat_interleave(num_samples)
        node_index = [
            torch.randperm(num_mono_node, device=self.device)[:num_sample.item()]
            for num_mono_node, num_sample in zip(num_nodes, num_samples)
        ]
        node_index = torch.cat(node_index, dim=0)
        node_index = node_index + (num_cum_nodes - num_nodes)[sample2graph]

        target = graph.unit_type[node_index]
        input = graph.node_feature.float()
        input[node_index] = 0

        output = self.model(graph, input, all_loss, metric)
        node_feature = output["node_feature"]
        node_feature = node_feature[node_index]
        pred = self.mlp(node_feature)

        return pred, target

    def evaluate(self, pred, target):
        metric = {}
        accuracy = (pred.argmax(dim=-1) == target).float().mean()

        name = tasks._get_metric_name("acc")
        metric[name] = accuracy

        return metric

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)
        metric.update(self.evaluate(pred, target))

        loss = F.cross_entropy(pred, target)
        name = tasks._get_criterion_name("ce")
        metric[name] = loss

        all_loss += loss

        return all_loss, metric

@R.register("tasks.SMPC")
class SMPC(tasks.Task, core.Configurable):

    mono_mask_token = Glycan.units.index("Unknown")
    atom_mask_token = constant.ATOM_SYMBOL.index("Null") + len(Glycan.units)

    def __init__(self, model, mask_rate=0.15, num_mlp_layer=2):
        super(SMPC, self).__init__()
        self.model = model
        self.mask_rate = mask_rate
        self.num_mlp_layer = num_mlp_layer

    def preprocess(self, train_set, valid_set, test_set):
        model_output_dim = self.model.output_dim
        num_label = 210
        self.mlp = layers.MLP(model_output_dim, [model_output_dim] * (self.num_mlp_layer - 1) + [num_label])

    def predict_and_target(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        num_nodes = graph.num_nodes
        num_cum_nodes = num_nodes.cumsum(0)
        mono_mask = graph.unit_type < len(Glycan.units)

        mono_mask_split = torch.split(mono_mask, num_nodes.tolist())
        num_mono_nodes = torch.tensor([sum(indices) for indices in mono_mask_split], device=self.device)
        num_samples = (num_mono_nodes * self.mask_rate).long().clamp(1)

        sample2graph = torch.repeat_interleave(num_samples)

        selected_mono_indices = [
            torch.randperm(num_mono_node)[:num_sample.item()]
            for num_mono_node, num_sample in zip(num_mono_nodes, num_samples)
        ]
        selected_mono_indices = torch.cat(selected_mono_indices, dim=0)

        atom_indices = list()
        for graph_id, mono_indice in zip(sample2graph, selected_mono_indices):
            atom_start = graph.cumsums[graph_id][mono_indice] + 1 + num_cum_nodes[graph_id] - num_nodes[graph_id]
            atom_end = graph.cumsums[graph_id][mono_indice + 1] + num_cum_nodes[graph_id] - num_nodes[graph_id]
            atom_indices.append(torch.arange(atom_start, atom_end, device=self.device))

        node_index = torch.cat(atom_indices, dim=0)

        target = graph.unit_type[node_index]
        input = graph.node_feature.float()
        graph.unit_type[node_index] = self.atom_mask_token

        output = self.model(graph, input, all_loss, metric)
        node_feature = output["node_feature"][node_index]
        pred = self.mlp(node_feature)

        return pred, target

    def evaluate(self, pred, target):
        metric = {}
        accuracy = (pred.argmax(dim=-1) == target).float().mean()

        name = tasks._get_metric_name("acc")
        metric[name] = accuracy

        return metric

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)
        metric.update(self.evaluate(pred, target))

        loss = F.cross_entropy(pred, target)
        name = tasks._get_criterion_name("ce")
        metric[name] = loss

        all_loss += loss

        return all_loss, metric


@R.register("tasks.HeteroContextPrediction")
class HeteroContextPrediction(tasks.Task, core.Configurable):
    """
    Implementation of Context prediction task proposed in `Strategies for Pre-training Graph Neural Networks`_ for heterogeneous graph.

    .. _Strategies for Pre-training Graph Neural Networks:
        https://arxiv.org/abs/1905.12265

    For a given center node, the subgraph is defined as a k-hop neighborhood (inclusive) around the selected node.
    The context graph is defined as the surrounding graph structure between r1- (exclusive) and r2-hop (inclusive)
    from the center node. Nodes between k- and r1-hop are picked as anchor nodes for the context representation.

    Parameters:
        model (nn.Module): node representation model for subgraphs.
        context_model (nn.Module, optional): node representation model for context graphs.
            By default, use the same architecture as ``model`` without parameter sharing.
        k (int, optional): radius for subgraphs
        r1 (int, optional): inner radius for context graphs
        r2 (int, optional): outer radius for context graphs
        readout (nn.Module, optional): readout function over context anchor nodes
        num_negative (int, optional): number of negative samples per positive sample
    """

    def __init__(self, model, context_model=None, k=5, r1=4, r2=7, readout="mean", num_negative=1):
        super(HeteroContextPrediction, self).__init__()
        self.model = model
        self.k = k
        self.r1 = r1
        self.r2 = r2
        self.num_negative = num_negative
        assert r1 < k < r2

        if context_model is None:
            self.context_model = copy.deepcopy(model)
        else:
            self.context_model = context_model
        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        else:
            raise ValueError("Unknown readout `%s`" % readout)
    
    def add_atom_mask(self, graph, mask, index2graph):
        masked_index = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        masked_index2graph = index2graph[masked_index]
        subindex = masked_index - (graph.num_cum_nodes - graph.num_nodes)[masked_index2graph]

        cumsums = graph.cumsums

        node_subindex_atom = []
        num_samples_atom = torch.zeros_like(graph.num_cum_nodes, dtype=torch.long)

        for graph_id, index in zip(masked_index2graph, subindex):
            node_subindex_atom.extend([
                x for x in range(cumsums[graph_id][index], cumsums[graph_id][index + 1])
            ])
            num_samples_atom[graph_id] += cumsums[graph_id][index + 1] - cumsums[graph_id][index]
        
        node_subindex_atom = torch.tensor(node_subindex_atom, dtype=torch.long, device=self.device)
        sample2graph_atom = torch.repeat_interleave(num_samples_atom)
        node_index_atom = node_subindex_atom + (graph.num_cum_nodes - graph.num_nodes)[sample2graph_atom]

        atom_mask = torch.zeros_like(mask, dtype=torch.bool, device=self.device)
        atom_mask[node_index_atom] = True
        combined_mask = mask | atom_mask

        return combined_mask

    def substruct_and_context(self, graph):
        num_monos = torch.tensor([row[0] for row in graph.cumsums], dtype=torch.long, device=self.device)
        center_index = (torch.rand(len(graph), device=self.device) * num_monos).long()
        center_index = center_index + graph.num_cum_nodes - graph.num_nodes
        dist = torch.full((graph.num_node,), self.r2 + 1, dtype=torch.long, device=self.device)
        dist[center_index] = 0

        node_in, node_out = graph.edge_list.t()[:2]
        node_in_type = graph.unit_type[node_in]
        node_out_type = graph.unit_type[node_out]
        mono_mask = (node_in_type < len(Glycan.units)) & (node_out_type < len(Glycan.units))
        mono_edge_list = graph.edge_list[mono_mask]

        mono_node_in, mono_node_out, _ = mono_edge_list.t()
        mono_node_map = torch.zeros(graph.num_node, dtype=torch.bool, device=self.device)
        mono_node_map[torch.unique(mono_node_in)] = True
        for _ in range(self.r2):
            new_dist = scatter_min(dist[mono_node_in], mono_node_out, dim_size=graph.num_node)[0] + 1
            dist[mono_node_map] = torch.min(dist[mono_node_map], new_dist[mono_node_map])
        
        index2graph = torch.repeat_interleave(graph.num_nodes)
        substruct_mask = self.add_atom_mask(graph, dist <= self.k, index2graph)
        context_mask = self.add_atom_mask(graph, (dist > self.r1) & (dist <= self.r2), index2graph)
        is_center_node = functional.as_mask(center_index, graph.num_node)
        is_anchor_node = (dist > self.r1) & (dist <= self.k)

        substruct = graph.clone()
        context = graph.clone()
        with substruct.node():
            substruct.is_center_node = is_center_node
        with context.node():
            context.is_anchor_node = is_anchor_node

        substruct = substruct.subgraph(substruct_mask)
        context = context.subgraph(context_mask)
        valid = context.num_nodes > 0
        substruct = substruct[valid]
        context = context[valid]

        return substruct, context

    def predict_and_target(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        substruct, context = self.substruct_and_context(graph)
        anchor = context.subgraph(context.is_anchor_node)

        substruct_output = self.model(substruct, substruct.node_feature.float(), all_loss, metric)
        substruct_feature = substruct_output["all_node_feature"][substruct.is_center_node]

        context_output = self.context_model(context, context.node_feature.float(), all_loss, metric)
        anchor_feature = context_output["all_node_feature"][context.is_anchor_node]
        context_feature = self.readout(anchor, anchor_feature)

        shift = torch.arange(self.num_negative, device=self.device) + 1
        neg_index = (torch.arange(len(context), device=self.device).unsqueeze(-1) + shift) % len(context)
        context_feature = torch.cat([context_feature.unsqueeze(1), context_feature[neg_index]], dim=1)
        substruct_feature = substruct_feature.unsqueeze(1).expand_as(context_feature)

        pred = torch.einsum("bnd, bnd -> bn", substruct_feature, context_feature)
        target = torch.zeros_like(pred)
        target[:, 0] = 1
        return pred, target

    def evaluate(self, pred, target):
        metric = {}
        accuracy = ((pred > 0) == (target > 0.5)).float().mean()

        name = tasks._get_metric_name("acc")
        metric[name] = accuracy

        return metric

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)
        metric.update(self.evaluate(pred, target))

        loss = F.binary_cross_entropy_with_logits(pred, target)
        name = tasks._get_criterion_name("bce")
        metric[name] = loss

        all_loss += loss

        return all_loss, metric

