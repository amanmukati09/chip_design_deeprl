# ml/predictor.py
# Loads trained CircuitGNN and predicts PAC cost.
#
# Changes vs old version:
#   1. Loads cost normalization stats (cost_stats.json)
#      to convert normalized predictions back to real PAC cost
#   2. predict() returns real PAC cost (not normalized)
#   3. Works with multi-circuit normalized training

import os
import sys
import time
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnn_model import load_samples, sample_to_tensors
from trainer import SmallCircuitGNN

MODEL_PATH    = "ml/models/gnn_trained.pt"
STATS_PATH    = "ml/models/cost_stats.json"
NODE_FEATURES = 5
HIDDEN_DIM    = 32
OUTPUT_DIM    = 1


class GNNPredictor:
    """Wraps trained CircuitGNN for fast cost prediction."""

    def __init__(self, model_path=MODEL_PATH, stats_path=STATS_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No model at {model_path}. Run ml/trainer.py first.")

        # self.model = CircuitGNN(node_features=NODE_FEATURES,
        #                          hidden_dim=HIDDEN_DIM,
        #                          output_dim=OUTPUT_DIM)
        self.model = SmallCircuitGNN(node_features=NODE_FEATURES, hidden_dim=HIDDEN_DIM)
        

        self.model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True))
        self.model.eval()

        # Load cost normalization stats if available
        self.cost_stats = {}
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                raw = json.load(f)
            # Keys stored as strings — convert back to int
            self.cost_stats = {int(k): tuple(v) for k, v in raw.items()}

        print(f"[Predictor] Model loaded from {model_path}")
        if self.cost_stats:
            print(f"[Predictor] Cost stats loaded for "
                  f"{len(self.cost_stats)} gate-count tiers")

    def _denormalize(self, norm_cost: float, gate_count: int) -> float:
        """Convert normalized prediction back to real PAC cost."""
        if not self.cost_stats:
            return norm_cost
        # Find nearest tier if exact not found
        if gate_count in self.cost_stats:
            mean, std = self.cost_stats[gate_count]
        else:
            nearest = min(self.cost_stats.keys(),
                          key=lambda k: abs(k - gate_count))
            mean, std = self.cost_stats[nearest]
        return norm_cost * std + mean

    def predict(self, node_features, edge_index,
                gate_count: int = 0) -> float:
        """
        Predict PAC cost for one circuit.

        Args:
            node_features : list of lists OR torch.Tensor [N, 3]
            edge_index    : list of [src, dst] pairs
            gate_count    : used for denormalization (pass circuit.gate_count)

        Returns:
            Predicted real PAC cost (float)
        """
        if not isinstance(node_features, torch.Tensor):
            node_features = torch.tensor(node_features, dtype=torch.float32)

        num_nodes = node_features.size(0)
        with torch.no_grad():
            pred = self.model(node_features, edge_index, num_nodes)

        norm_cost = pred.item()
        return round(self._denormalize(norm_cost, gate_count), 4)

    def predict_from_sample(self, sample: dict):
        """Predict directly from a JSON training sample dict."""
        node_feat, edge_index, true_cost = sample_to_tensors(sample)
        gc        = sample.get('gate_count', 0)
        predicted = self.predict(node_feat, edge_index, gate_count=gc)
        # true_cost in sample may be normalized — use cost_raw if available
        real_cost = sample.get('cost_raw', true_cost.item())
        return predicted, real_cost


if __name__ == "__main__":
    SAMPLES_PATH = "ml/data/multi_circuit_samples.json"
    if not os.path.exists(SAMPLES_PATH):
        SAMPLES_PATH = "ml/data/s1196_samples.json"

    print("=" * 58)
    print("  GNN Predictor Test")
    print("=" * 58)

    predictor = GNNPredictor()
    samples   = load_samples(SAMPLES_PATH)

    # Test on 10 samples
    print(f"\n  {'#':>4}  {'Gates':>6}  {'True':>10}  "
          f"{'Predicted':>10}  {'Error%':>8}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}")

    errors = []
    for i, sample in enumerate(samples[:10]):
        predicted, true_cost = predictor.predict_from_sample(sample)
        gc  = sample.get('gate_count', 0)
        err = abs(predicted - true_cost) / max(true_cost, 1) * 100
        errors.append(err)
        print(f"  {i+1:>4}  {gc:>6}  {true_cost:>10.2f}  "
              f"{predicted:>10.2f}  {err:>7.2f}%")

    print(f"\n  Avg error : {sum(errors)/len(errors):.2f}%")

    # Speed benchmark
    times = []
    for sample in samples[:50]:
        node_feat, edge_index, _ = sample_to_tensors(sample)
        gc = sample.get('gate_count', 0)
        t0 = time.perf_counter()
        predictor.predict(node_feat, edge_index, gate_count=gc)
        times.append(time.perf_counter() - t0)
    print(f"  Avg speed : {sum(times)/len(times)*1000:.2f} ms/circuit")
    print("=" * 58)