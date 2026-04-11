# ml/gnn_model.py
# Graph Neural Network for PAC cost prediction
#
# Architecture:
#   Input  → node features (gate_type, fan_in, fan_out)
#   Layer1 → GraphSAGE convolution (learns local structure)
#   Layer2 → GraphSAGE convolution (learns global structure)
#   Pool   → mean pooling (circuit level embedding)
#   Output → predicted PAC cost
#
# Why GraphSAGE?
#   - Works without pretraining
#   - Handles variable size graphs (different circuits)
#   - Generalizes to unseen circuits
#   - Used in Google AlphaChip research

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# ─────────────────────────────────────────────────────────────
# GRAPH SAGE CONVOLUTION LAYER
# Manual implementation — no PyTorch Geometric needed
# Works with our existing data format
# ─────────────────────────────────────────────────────────────

class SAGEConv(nn.Module):
    """
    GraphSAGE Convolution Layer.

    For each node:
        1. Aggregate neighbor features (mean)
        2. Concatenate with own features
        3. Apply linear transformation
        4. Apply activation

    This lets each node learn from its neighborhood.
    After 2 layers, each node knows about 2-hop neighbors.
    """

    def __init__(self, in_features, out_features):
        super(SAGEConv, self).__init__()
        # Transform self + neighbor features
        self.linear = nn.Linear(in_features * 2, out_features)
        self.norm   = nn.BatchNorm1d(out_features)

    def forward(self, x, edge_index, num_nodes):
        """
        Args:
            x          : node features [num_nodes, in_features]
            edge_index : list of [src, dst] pairs
            num_nodes  : total number of nodes

        Returns:
            out : updated node features [num_nodes, out_features]
        """
        # ── Aggregate neighbor features ──────────────
        # For each node, compute mean of neighbor features
        neighbor_sum   = torch.zeros(num_nodes, x.size(1),
                                     device=x.device)
        neighbor_count = torch.zeros(num_nodes, 1,
                                     device=x.device)

        if len(edge_index) > 0:
            # Convert edge_index to tensor
            edges  = torch.tensor(edge_index,
                                  dtype=torch.long,
                                  device=x.device)
            src    = edges[:, 0]
            dst    = edges[:, 1]

            # Scatter add neighbor features to dst nodes
            neighbor_sum.index_add_(0, dst, x[src])
            ones = torch.ones(dst.size(0), 1, device=x.device)
            neighbor_count.index_add_(0, dst, ones)

        # Mean aggregation
        neighbor_count = neighbor_count.clamp(min=1)
        neighbor_mean  = neighbor_sum / neighbor_count

        # ── Concatenate self + neighbor ───────────────
        combined = torch.cat([x, neighbor_mean], dim=1)

        # ── Transform ────────────────────────────────
        out = self.linear(combined)
        out = self.norm(out)
        out = F.relu(out)

        return out


# ─────────────────────────────────────────────────────────────
# GNN MODEL
# ─────────────────────────────────────────────────────────────

class CircuitGNN(nn.Module):
    """
    Graph Neural Network for circuit PAC cost prediction.

    Architecture:
        SAGEConv(3  → 64)   learns gate-level patterns
        SAGEConv(64 → 128)  learns circuit-level patterns
        SAGEConv(128→ 64)   compress representation
        MeanPool            circuit-level embedding
        Linear(64 → 32)     feature reduction
        Linear(32 → 1)      cost prediction
    """

    def __init__(self,
                 node_features = 3,    # gate_type, fan_in, fan_out
                 hidden_dim    = 64,
                 output_dim    = 1):   # predicted cost
        super(CircuitGNN, self).__init__()

        # Graph convolution layers
        self.conv1 = SAGEConv(node_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim * 2)
        self.conv3 = SAGEConv(hidden_dim * 2, hidden_dim)

        # Prediction head
        self.fc1   = nn.Linear(hidden_dim, 32)
        self.fc2   = nn.Linear(32, output_dim)
        self.drop  = nn.Dropout(0.2)

    def forward(self, node_features, edge_index, num_nodes):
        """
        Args:
            node_features : tensor [num_nodes, 3]
            edge_index    : list of [src, dst] pairs
            num_nodes     : int

        Returns:
            cost_pred : predicted PAC cost (scalar)
        """
        x = node_features

        # Graph convolutions
        x = self.conv1(x, edge_index, num_nodes)
        x = self.drop(x)
        x = self.conv2(x, edge_index, num_nodes)
        x = self.drop(x)
        x = self.conv3(x, edge_index, num_nodes)

        # Global mean pooling
        # Compress all node embeddings into one circuit embedding
        x = x.mean(dim=0, keepdim=True)

        # Prediction
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x.squeeze()


# ─────────────────────────────────────────────────────────────
# DATA LOADING UTILITIES
# ─────────────────────────────────────────────────────────────

def load_samples(filepath):
    """Loads training samples from JSON file."""
    with open(filepath, 'r') as f:
        samples = json.load(f)
    print(f"[GNN] Loaded {len(samples)} samples from {filepath}")
    return samples


def sample_to_tensors(sample):
    """
    Converts one JSON sample to PyTorch tensors.

    Returns:
        node_feat  : tensor [num_nodes, 3]
        edge_index : list of [src, dst]
        cost       : tensor scalar
    """
    node_feat  = torch.tensor(
        sample['node_features'],
        dtype=torch.float32
    )
    edge_index = sample['edge_index']
    cost       = torch.tensor(
        sample['cost'],
        dtype=torch.float32
    )
    return node_feat, edge_index, cost


# ── Quick test ───────────────────────────────────────
if __name__ == "__main__":

    print("Testing GNN architecture...")

    # Create model
    model = CircuitGNN(
        node_features = 3,
        hidden_dim    = 64,
        output_dim    = 1
    )

    print(f"\nModel architecture:")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Test with one sample from our data
    sample_path = "ml/data/s1196_samples.json"
    if os.path.exists(sample_path):
        samples   = load_samples(sample_path)
        sample    = samples[0]

        node_feat, edge_index, cost = sample_to_tensors(sample)

        print(f"\nSample info:")
        print(f"  Node features shape : {node_feat.shape}")
        print(f"  Edges               : {len(edge_index)}")
        print(f"  True cost           : {cost.item()}")

        # Forward pass
        model.eval()
        with torch.no_grad():
            pred = model(node_feat, edge_index, node_feat.size(0))

        print(f"  Predicted cost      : {round(pred.item(), 4)}")
        print(f"\nGNN forward pass successful.")
        print(f"(Prediction is random before training — this is expected)")
    else:
        print("\nNo sample data found.")
        print("Run ml/data_collector.py first.")