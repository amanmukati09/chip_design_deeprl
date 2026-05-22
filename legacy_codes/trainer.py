# ml/trainer.py
# Fixed: adds weight decay + dropout increase to prevent overfitting.
# Best val loss was epoch 1 previously — model was memorizing, not learning.
#
# Fixes:
#   1. weight_decay=1e-4 in Adam (L2 regularization)
#   2. Dropout raised to 0.3 in CircuitGNN (defined inline here)
#   3. Early stopping — stop if val loss doesn't improve for 20 epochs
#   4. Batch accumulation: accumulate gradients over 8 samples
#      to simulate larger effective batch size

import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnn_model import SAGEConv, load_samples, sample_to_tensors

SAMPLES_PATH  = "ml/data/multi_circuit_samples.json"
MODEL_DIR     = "ml/models"
MODEL_PATH    = os.path.join(MODEL_DIR, "gnn_trained.pt")
LOG_PATH      = os.path.join(MODEL_DIR, "train_log.json")
STATS_PATH    = os.path.join(MODEL_DIR, "cost_stats.json")

EPOCHS          = 150
LR              = 5e-4       # lower LR — was converging too fast
WEIGHT_DECAY    = 1e-4       # L2 regularization
LR_DECAY_STEP   = 40
LR_DECAY_RATE   = 0.5
TRAIN_SPLIT     = 0.8
SEED            = 42
PATIENCE        = 25         # early stopping patience
GRAD_ACCUM      = 8          # accumulate gradients over N samples
NODE_FEATURES   = 3
HIDDEN_DIM      = 64
OUTPUT_DIM      = 1


# ─────────────────────────────────────────────────────────────
# REGULARIZED GNN — higher dropout than gnn_model.py default
# ─────────────────────────────────────────────────────────────

class RegularizedCircuitGNN(nn.Module):
    """Same architecture as CircuitGNN but dropout=0.3."""

    def __init__(self, node_features=3, hidden_dim=64, output_dim=1):
        super().__init__()
        self.conv1 = SAGEConv(node_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim * 2)
        self.conv3 = SAGEConv(hidden_dim * 2, hidden_dim)
        self.fc1   = nn.Linear(hidden_dim, 32)
        self.fc2   = nn.Linear(32, output_dim)
        self.drop  = nn.Dropout(0.3)   # was 0.2

    def forward(self, node_features, edge_index, num_nodes):
        x = node_features
        x = self.conv1(x, edge_index, num_nodes)
        x = self.drop(x)
        x = self.conv2(x, edge_index, num_nodes)
        x = self.drop(x)
        x = self.conv3(x, edge_index, num_nodes)
        x = x.mean(dim=0, keepdim=True)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze()


# ─────────────────────────────────────────────────────────────
# COST NORMALIZATION
# ─────────────────────────────────────────────────────────────

def normalize_costs(samples: list):
    groups = {}
    for s in samples:
        groups.setdefault(s['gate_count'], []).append(s['cost'])

    stats = {}
    for gc, costs in groups.items():
        mean = sum(costs) / len(costs)
        var  = sum((c - mean) ** 2 for c in costs) / max(len(costs)-1, 1)
        std  = max(var ** 0.5, 1.0)
        stats[gc] = (mean, std)

    normalized = []
    for s in samples:
        ns         = dict(s)
        mean, std  = stats[s['gate_count']]
        ns['cost_raw'] = s['cost']
        ns['cost']     = (s['cost'] - mean) / std
        normalized.append(ns)

    return normalized, stats


def split_data(samples, ratio=0.8, seed=42):
    random.seed(seed)
    data = samples[:]
    random.shuffle(data)
    cut  = int(len(data) * ratio)
    return data[:cut], data[cut:]


# ─────────────────────────────────────────────────────────────
# TRAINING WITH GRADIENT ACCUMULATION
# ─────────────────────────────────────────────────────────────

def run_epoch(model, samples, optimizer, criterion,
               training=True, grad_accum=1):
    total = 0.0
    model.train() if training else model.eval()

    if training:
        optimizer.zero_grad()

    for i, sample in enumerate(samples):
        node_feat, edge_index, true_cost = sample_to_tensors(sample)

        with torch.set_grad_enabled(training):
            pred = model(node_feat, edge_index, node_feat.size(0))
            loss = criterion(pred, true_cost)
            if training:
                (loss / grad_accum).backward()

        total += loss.item()

        if training and (i + 1) % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()

    # Flush remaining gradients
    if training and len(samples) % grad_accum != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total / max(len(samples), 1)


def train(samples_path: str = SAMPLES_PATH):
    print("=" * 62)
    print("  CircuitGNN Trainer — Regularized")
    print("=" * 62)

    if not os.path.exists(samples_path):
        print(f"\n[ERROR] Not found: {samples_path}")
        print("Run: python ml/data_collector.py")
        return

    raw = load_samples(samples_path)
    print(f"\n  Raw samples : {len(raw)}")

    from collections import Counter
    dist = Counter(s['gate_count'] for s in raw)
    print(f"  By circuit  : {dict(sorted(dist.items()))}")

    samples, cost_stats = normalize_costs(raw)
    train_set, val_set  = split_data(samples, TRAIN_SPLIT, SEED)
    print(f"  Train/Val   : {len(train_set)} / {len(val_set)}")

    model     = RegularizedCircuitGNN(NODE_FEATURES, HIDDEN_DIM, OUTPUT_DIM)
    opt       = torch.optim.Adam(model.parameters(),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(
        opt, step_size=LR_DECAY_STEP, gamma=LR_DECAY_RATE)
    criterion = nn.MSELoss()

    print(f"  Params      : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  LR          : {LR}  weight_decay={WEIGHT_DECAY}")
    print(f"  Dropout     : 0.3  grad_accum={GRAD_ACCUM}")
    print(f"  Patience    : {PATIENCE} epochs\n")
    print(f"  {'Epoch':>5}  {'Train':>8}  {'Val':>8}  {'Note'}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*12}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    log        = {"train_loss": [], "val_loss": []}
    best_val   = float("inf")
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        tl = run_epoch(model, train_set, opt, criterion,
                        training=True, grad_accum=GRAD_ACCUM)
        vl = run_epoch(model, val_set, opt, criterion,
                        training=False)
        scheduler.step()

        log["train_loss"].append(round(tl, 4))
        log["val_loss"].append(round(vl, 4))

        if vl < best_val:
            best_val   = vl
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
            note = "← best"
        else:
            no_improve += 1
            note = ""

        if epoch % 10 == 0 or epoch == 1 or note:
            print(f"  {epoch:5d}  {tl:>8.4f}  {vl:>8.4f}  {note}")

        if no_improve >= PATIENCE:
            print(f"\n  Early stop at epoch {epoch} "
                  f"(no improvement for {PATIENCE} epochs)")
            break

    with open(LOG_PATH, 'w') as f:
        json.dump(log, f, indent=2)

    serial = {str(k): list(v) for k, v in cost_stats.items()}
    with open(STATS_PATH, 'w') as f:
        json.dump(serial, f, indent=2)

    print(f"\n  Best val : {best_val:.4f}  (epoch {best_epoch})")
    print(f"  Model    : {MODEL_PATH}")
    print(f"  Stats    : {STATS_PATH}")
    print("=" * 62)
    print("\nNext: python ml/predictor.py")


if __name__ == "__main__":
    train()
