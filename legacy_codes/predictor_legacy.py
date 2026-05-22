"""
ml/predictor.py
─────────────────────────────────────────────────────
Loads the trained CircuitGNN and predicts PAC cost
for any circuit — fast, no optimization needed.

Two modes:
  1. Used as a module  → GNNPredictor class
  2. Run directly      → python ml/predictor.py
                         (tests on a sample and compares to true cost)
"""

import os
import sys
import time
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnn_model import CircuitGNN, load_samples, sample_to_tensors


# ─────────────────────────────────────────────────────────────
# CONFIG  (must match trainer.py exactly)
# ─────────────────────────────────────────────────────────────

MODEL_PATH    = "ml/models/gnn_trained.pt"
NODE_FEATURES = 3
HIDDEN_DIM    = 64
OUTPUT_DIM    = 1


# ─────────────────────────────────────────────────────────────
# PREDICTOR CLASS
# ─────────────────────────────────────────────────────────────

class GNNPredictor:
    """
    Wraps the trained CircuitGNN for fast cost prediction.

    Usage:
        predictor = GNNPredictor()
        cost = predictor.predict(node_features, edge_index)
    """

    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model at: {model_path}\n"
                f"Run ml/trainer.py first."
            )

        self.model = CircuitGNN(
            node_features=NODE_FEATURES,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        self.model.eval()
        print(f"[Predictor] Model loaded from {model_path}")

    def predict(self, node_features, edge_index):
        """
        Predict PAC cost for one circuit.

        Args:
            node_features : list of lists  OR  torch.Tensor [N, 3]
            edge_index    : list of [src, dst] pairs

        Returns:
            cost : float  (predicted PAC cost)
        """
        if not isinstance(node_features, torch.Tensor):
            node_features = torch.tensor(node_features, dtype=torch.float32)

        num_nodes = node_features.size(0)

        with torch.no_grad():
            pred = self.model(node_features, edge_index, num_nodes)

        return round(pred.item(), 4)

    def predict_from_sample(self, sample):
        """Convenience: predict directly from a JSON sample dict."""
        node_feat, edge_index, true_cost = sample_to_tensors(sample)
        predicted = self.predict(node_feat, edge_index)
        return predicted, true_cost.item()


# ─────────────────────────────────────────────────────────────
# SPEED BENCHMARK  (how fast vs the real cost function?)
# ─────────────────────────────────────────────────────────────

def benchmark_speed(predictor, samples, n=50):
    """Run n predictions and measure average time per prediction."""
    times = []
    for sample in samples[:n]:
        node_feat, edge_index, _ = sample_to_tensors(sample)
        t0 = time.perf_counter()
        predictor.predict(node_feat, edge_index)
        times.append(time.perf_counter() - t0)

    avg_ms = (sum(times) / len(times)) * 1000
    return avg_ms


# ─────────────────────────────────────────────────────────────
# MAIN  — test the predictor
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    SAMPLES_PATH = "ml/data/s1196_samples.json"

    print("=" * 58)
    print("  GNN Predictor Test")
    print("=" * 58)

    # 1. Load predictor
    predictor = GNNPredictor()

    # 2. Load samples
    samples = load_samples(SAMPLES_PATH)

    # 3. Test on 10 samples — predicted vs true
    print(f"\n{'Sample':>8}  {'True Cost':>10}  {'Predicted':>10}  {'Error':>8}  {'Error %':>8}")
    print("-" * 55)

    errors = []
    for i, sample in enumerate(samples[:10]):
        predicted, true_cost = predictor.predict_from_sample(sample)
        error    = abs(predicted - true_cost)
        error_pct = (error / true_cost) * 100
        errors.append(error_pct)
        print(f"  {i+1:5d}    {true_cost:>10.2f}  {predicted:>10.2f}  {error:>8.2f}  {error_pct:>7.2f}%")

    avg_error = sum(errors) / len(errors)
    print(f"\n  Average error across 10 samples: {avg_error:.2f}%")

    # 4. Speed benchmark
    avg_ms = benchmark_speed(predictor, samples, n=50)
    print(f"  Average prediction time        : {avg_ms:.3f} ms per circuit")
    print()
    print("=" * 58)
    print("  Predictor working!")
    print("  Ready to plug into optimizer as fast cost estimator.")
    print("=" * 58)
    print("\nNext step: plug GNNPredictor into hybrid_optimizer.py")