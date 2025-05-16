import torch
from torch import nn

from torchdrug import core
from torchdrug.core import Registry as R
from torch_scatter import scatter_add


@R.register("layers.SAM")
class SparseAttentionModule(nn.Module, core.Configurable):
    """
    Sparse Attention Module (SAM) implementation based on the atom-level and residue-level graphs.
    
    Parameters:
        input_dim (int): Input dimension of node features
        output_dim (int): Output dimension for node features
        edge_dim (int): Dimension of edge features
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate for attention scores
    """
    
    def __init__(self, input_dim, output_dim, edge_dim, num_heads=4, dropout=0.1):
        super(SparseAttentionModule, self).__init__()
        
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.head_dim = output_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        
        self.query_proj = nn.Linear(input_dim, output_dim)
        self.key_proj = nn.Linear(input_dim, output_dim)
        self.value_proj = nn.Linear(input_dim, output_dim)
        self.edge_proj = nn.Linear(edge_dim, output_dim)

    def forward(self, graph, node_features, edge_features):
        """
        Forward pass for the Sparse Attention Module.
        
        Parameters:
            graph (Graph): The graph structure with edge information
            node_features (Tensor): Node features of shape (num_nodes, input_dim)
            edge_features (Tensor): Edge features of shape (num_edges, edge_dim)
        
        Returns:
            Tensor: Updated node features of shape (num_nodes, output_dim)
        """
        Q = self.query_proj(node_features).view(-1, self.num_heads, self.head_dim)
        K = self.key_proj(node_features).view(-1, self.num_heads, self.head_dim)
        V = self.value_proj(node_features).view(-1, self.num_heads, self.head_dim)
        
        edge_bias = self.edge_proj(edge_features).view(-1, self.num_heads, self.head_dim)

        node_in, node_out = graph.edge_list[:, 0], graph.edge_list[:, 1]
        
        attention_scores = (Q[node_in] * K[node_out]).sum(dim=-1) * self.scale
        attention_scores += edge_bias.sum(dim=-1)
        
        attention_weights = torch.softmax(attention_scores, dim=1)
        attention_weights = self.dropout(attention_weights)
        
        node_out_features = V[node_in] * attention_weights.unsqueeze(-1)
        
        node_updates = scatter_add(node_out_features, node_out, dim=0, dim_size=graph.num_node)
        
        return node_updates.view(-1, self.output_dim)

@R.register("layers.SAMBlock")
class SAMBlock(nn.Module, core.Configurable):
    def __init__(self, atom_dim, mono_dim, edge_dim, num_heads, dropout=0.1):
        super(SAMBlock, self).__init__()

        self.atom_sam = SparseAttentionModule(atom_dim, atom_dim, edge_dim, num_heads, dropout)
        self.mono_sam = SparseAttentionModule(mono_dim, mono_dim, edge_dim, num_heads, dropout)
        
        self.atom_ffn = nn.Sequential(
            nn.Linear(atom_dim, atom_dim * 2),
            nn.ReLU(),
            nn.Linear(atom_dim * 2, atom_dim)
        )
        self.mono_ffn = nn.Sequential(
            nn.Linear(mono_dim, mono_dim * 2),
            nn.ReLU(),
            nn.Linear(mono_dim * 2, mono_dim)
        )
        
    def forward(self, atom_graph, atom_input, mono_graph, mono_input):
        """ Implement by author's figure 2. """

        atom_edge_features = atom_graph.edge_feature
        mono_edge_features = mono_graph.edge_feature

        atom_out = self.atom_sam(atom_graph, atom_input.float(), atom_edge_features.float())
        atom_out = self.atom_ffn(atom_out) + atom_input

        mono_out = self.mono_sam(mono_graph, mono_input.float(), mono_edge_features.float())
        mono_out = self.mono_ffn(mono_out) + mono_input

        return atom_out, mono_out