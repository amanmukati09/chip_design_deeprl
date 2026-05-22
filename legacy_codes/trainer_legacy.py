"""
ml/trainer.py
─────────────────────────────────────────────────────
Trains CircuitGNN on samples collected by data_collector.py.

Usage (from project root):
    python ml/trainer.py

Output:
    ml/models/gnn_trained.pt   ← best model weights
    ml/models/train_log.json   ← loss history
"""

import os
import json
import random
import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # adds ml/ to path
from gnn_model import CircuitGNN, load_samples, sample_to_tensors


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

SAMPLES_PATH = "ml/data/s1196_samples_v2.json"
MODEL_DIR    = "ml/models"
MODEL_PATH   = os.path.join(MODEL_DIR, "gnn_trained.pt")
LOG_PATH     = os.path.join(MODEL_DIR, "train_log.json")

EPOCHS       = 50
LR           = 1e-3
TRAIN_SPLIT  = 0.8
SEED         = 42

# Must match gnn_model.py — confirmed from model output:
# SAGEConv(in_features=6, ...) → node features = 6
NODE_FEATURES = 3
HIDDEN_DIM    = 64
OUTPUT_DIM    = 1


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def split_data(samples, train_ratio=0.8, seed=42):
    random.seed(seed)
    data = samples[:]
    random.shuffle(data)
    cut = int(len(data) * train_ratio)
    return data[:cut], data[cut:]


def run_epoch(model, samples, optimizer, criterion, training=True):
    """
    One full pass over samples.
    Batch size = 1 (variable graph sizes require per-sample forward pass).
    Returns average loss.
    """
    total_loss = 0.0

    if training:
        model.train()
    else:
        model.eval()

    for sample in samples:
        node_feat, edge_index, true_cost = sample_to_tensors(sample)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            pred = model(node_feat, edge_index, node_feat.size(0))
            loss = criterion(pred, true_cost)

        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(samples)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def train():
    print("=" * 58)
    print("  CircuitGNN Trainer")
    print("=" * 58)

    # 1. Load data
    if not os.path.exists(SAMPLES_PATH):
        print(f"\n[ERROR] Samples not found: {SAMPLES_PATH}")
        print("Run ml/data_collector.py first.")
        return

    samples = load_samples(SAMPLES_PATH)
    train_set, val_set = split_data(samples, TRAIN_SPLIT, SEED)

    print(f"\nDataset      : {len(samples)} total samples")
    print(f"Train        : {len(train_set)}  |  Val: {len(val_set)}")

    # 2. Model
    model = CircuitGNN(
        node_features=NODE_FEATURES,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters   : {total_params:,}")
    print(f"Epochs       : {EPOCHS}  |  LR: {LR}")
    print()
    print(f"{'Epoch':>7}  {'Train Loss':>12}  {'Val Loss':>10}  {'':>8}")
    print("-" * 50)

    # 3. Training loop
    log = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    os.makedirs(MODEL_DIR, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_set, optimizer, criterion, training=True)
        val_loss   = run_epoch(model, val_set,   optimizer, criterion, training=False)

        log["train_loss"].append(round(train_loss, 4))
        log["val_loss"].append(round(val_loss,   4))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            marker = "<- best saved"
        else:
            marker = ""

        print(f"  {epoch:3d}/{EPOCHS}  {train_loss:>12.2f}  {val_loss:>10.2f}  {marker}")

    # 4. Save log
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

    # 5. Summary
    print()
    print("=" * 58)
    print("  Training complete!")
    print(f"  Best val loss : {best_val_loss:.2f}")
    print(f"  Model saved   : {MODEL_PATH}")
    print(f"  Log saved     : {LOG_PATH}")
    print("=" * 58)
    print("\nNext step: python ml/predictor.py")


if __name__ == "__main__":
    train()