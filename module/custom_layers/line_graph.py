import torch
from torch import nn

from torchdrug import core
from torchdrug.core import Registry as R


@R.register("layers.SimpleSpatialLineGraph")
class SimpleSpatialLineGraph(nn.Module, core.Configurable):
    """
    Simple implementation of Spatial line graph construction module from `Protein Representation Learning by Geometric Structure Pretraining`_.

    .. _Protein Representation Learning by Geometric Structure Pretraining:
        https://arxiv.org/pdf/2203.06125.pdf

    All degrees will be count to 2/3 pi.
    Parameters:
        num_angle_bin (int, optional): number of bins to discretize angles between edges
    """

    def __init__(self, num_angle_bin=8):
        super(SimpleSpatialLineGraph, self).__init__()
        self.num_angle_bin = num_angle_bin
        self.fixed_relation = int((120 / 180) * (self.num_angle_bin / 2))

    def forward(self, graph):
        """
        Generate the spatial line graph of the input graph.
        The edge types are decided by the angles between two adjacent edges in the input graph.

        Parameters:
            graph (PackedGraph): :math:`n` graph(s)

        Returns:
            graph (PackedGraph): the spatial line graph
        """
        line_graph = graph.line_graph()
        relation = torch.full(
            (line_graph.edge_list.size(0),),
            self.fixed_relation,
            dtype=torch.long,
            device=line_graph.edge_list.device
        )
        edge_list = torch.cat([line_graph.edge_list, relation.unsqueeze(-1)], dim=-1)

        return type(line_graph)(edge_list, num_nodes=line_graph.num_nodes, offsets=line_graph._offsets,
                                num_edges=line_graph.num_edges, num_relation=self.num_angle_bin,
                                meta_dict=line_graph.meta_dict, **line_graph.data_dict)
