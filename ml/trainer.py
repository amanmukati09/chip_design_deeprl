# ml/trainer.py — circuit-level split, delta prediction, hidden_dim=32

import os, json, random, torch, torch.nn as nn, torch.nn.functional as F, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnn_model import SAGEConv, load_samples, sample_to_tensors
from sklearn.metrics import (
    mean_absolute_error,
    r2_score
)
from scipy.stats import spearmanr
import numpy as np


SAMPLES_PATH = "ml/data/multi_circuit_samples.json"
MODEL_DIR    = "ml/models"
MODEL_PATH   = os.path.join(MODEL_DIR, "gnn_trained.pt")
LOG_PATH     = os.path.join(MODEL_DIR, "train_log.json")
STATS_PATH   = os.path.join(MODEL_DIR, "cost_stats.json")

EPOCHS       = 100
LR           = 3e-5
WEIGHT_DECAY = 1e-5
PATIENCE     = 50
NODE_FEATURES = 5
HIDDEN_DIM   = 32   # reduced from 64
OUTPUT_DIM   = 1

# Held-out validation circuits — never seen during training
VAL_GATE_COUNTS = {659, 880, 1193, 2574}   # s1488=659, c1908=880
# Everything else trains


class SmallCircuitGNN(nn.Module):
    def __init__(self, node_features=4, hidden_dim=64):
        super().__init__()
        self.conv1 = SAGEConv(node_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        # self.fc1   = nn.Linear(hidden_dim*2, 32)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        # self.fc2   = nn.Linear(32, 1)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.drop  = nn.Dropout(0.3)

    def forward(self, node_features, edge_index, num_nodes):
        x = node_features
        # x = self.conv1(x, edge_index, num_nodes)
        # x = self.drop(x)
        # x = self.conv2(x, edge_index, num_nodes)

        x1 = self.conv1(x, edge_index, num_nodes)
        x1 = self.drop(x1)

        x2 = self.conv2(x1, edge_index, num_nodes)

        x = x1 + x2

        # x = x.mean(dim=0, keepdim=True)
        x_mean = x.mean(dim=0, keepdim=True)
        x_max  = x.max(dim=0, keepdim=True)[0]
        x = torch.cat([x_mean, x_max], dim=1)

        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze()


def normalize_costs(samples):
    groups = {}
    for s in samples:
        groups.setdefault(s['gate_count'], []).append(s['cost'])
    stats = {}
    for gc, costs in groups.items():
        mean = sum(costs)/len(costs)
        std  = max((sum((c-mean)**2 for c in costs)/max(len(costs)-1,1))**0.5, 1.0)
        stats[gc] = (mean, std)
    normalized = []
    for s in samples:
        ns = dict(s)
        mean, std = stats[s['gate_count']]
        ns['cost_raw'] = s['cost']
        ns['cost']     = (s['cost'] - mean) / std
        normalized.append(ns)
    return normalized, stats


def circuit_level_split(samples):
    """Split by circuit identity, not randomly."""
    train_set = [s for s in samples if s['gate_count'] not in VAL_GATE_COUNTS]
    val_set   = [s for s in samples if s['gate_count'] in VAL_GATE_COUNTS]
    return train_set, val_set


def run_epoch(model, samples, optimizer, criterion, training=True):
    total = 0.0
    model.train() if training else model.eval()
    random.shuffle(samples) if training else None
    for sample in samples:
        node_feat, edge_index, true_cost = sample_to_tensors(sample)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            pred = model(node_feat, edge_index, node_feat.size(0))
            loss = criterion(pred, true_cost)
        if training:
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += loss.item()
    return total / max(len(samples), 1)

def evaluate_metrics(model, samples):
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for sample in samples:
            node_feat, edge_index, true_cost = sample_to_tensors(sample)

            pred = model(
                node_feat,
                edge_index,
                node_feat.size(0)
            )

            preds.append(pred.item())
            targets.append(true_cost.item())

    preds = np.array(preds)
    targets = np.array(targets)

    mse = np.mean((preds - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)

    spear = spearmanr(targets, preds).correlation

    return {
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "spearman": round(spear, 4)
    }

def train():
    print("=" * 55)
    print("  CircuitGNN — Circuit-Level Split")
    print("=" * 55)

    if not os.path.exists(SAMPLES_PATH):
        print(f"[ERROR] Run data_collector.py first")
        return

    raw = load_samples(SAMPLES_PATH)
    samples, cost_stats = normalize_costs(raw)
    train_set, val_set  = circuit_level_split(samples)

    from collections import Counter
    print(f"  Train circuits: {dict(sorted(Counter(s['gate_count'] for s in train_set).items()))}")
    print(f"  Val circuits  : {dict(sorted(Counter(s['gate_count'] for s in val_set).items()))}")
    print(f"  Train/Val     : {len(train_set)} / {len(val_set)}")

    model     = SmallCircuitGNN(NODE_FEATURES, HIDDEN_DIM)
    opt       = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    criterion = nn.MSELoss()

    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}  hidden={HIDDEN_DIM}")
    print(f"\n  {'Epoch':>5}  {'Train':>8}  {'Val':>8}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*8}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    log = {"train_loss": [], "val_loss": []}
    best_val, best_epoch, no_improve = float("inf"), 0, 0

    for epoch in range(1, EPOCHS + 1):
        tl = run_epoch(model, train_set, opt, criterion, training=True)
        vl = run_epoch(model, val_set,   opt, criterion, training=False)
        scheduler.step()
        log["train_loss"].append(round(tl, 4))
        log["val_loss"].append(round(vl, 4))

        if vl < best_val:
            best_val, best_epoch, no_improve = vl, epoch, 0
            torch.save(model.state_dict(), MODEL_PATH)
            note = "← best"
        else:
            no_improve += 1
            note = ""

        # if epoch % 10 == 0 or epoch == 1 or note:
        print(f"  {epoch:5d}  {tl:>8.4f}  {vl:>8.4f}  {note}")

        if no_improve >= PATIENCE:
            print(f"\n  Early stop at epoch {epoch}")
            break

    with open(LOG_PATH, 'w') as f: json.dump(log, f, indent=2)
    serial = {str(k): list(v) for k, v in cost_stats.items()}
    with open(STATS_PATH, 'w') as f: json.dump(serial, f, indent=2)

    print(f"\n  Best val: {best_val:.4f} (epoch {best_epoch})")

    metrics = evaluate_metrics(model, val_set)

    print("\n  Validation Metrics")
    print("  ------------------")
    print(f"  MAE       : {metrics['mae']}")
    print(f"  RMSE      : {metrics['rmse']}")
    print(f"  R²        : {metrics['r2']}")
    print(f"  Spearman  : {metrics['spearman']}")

    print(f"  Model   : {MODEL_PATH}")
    print("=" * 55)

if __name__ == "__main__":
    print("WD: ", WEIGHT_DECAY, " LR: " , LR)
    train()