# agent/benchmark_ppo_hybrid.py
# Statistical validation: SA vs PPO vs PPO+SA hybrid.
# Runs N_SEEDS repetitions per circuit, reports mean +/- std.
# This is the results table for the paper.
#
# Usage:
#   python agent/benchmark_ppo_hybrid.py            (default circuit set)
#   python agent/benchmark_ppo_hybrid.py --all       (all available circuits)
#   python agent/benchmark_ppo_hybrid.py --seeds 10  (more seeds, tighter CI)

import sys, os, time, json, random, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch

from ml.predictor                  import GNNPredictor
from environment.chip_env          import ChipEnv
from agent.ppo_agent                import PPOAgent
from optimizer.simulated_annealing import simulated_annealing
from core.circuit                  import Circuit
from core.graph_builder            import build_graph
from core.feature_extractor        import extract_features
from core.pipeline                 import load_circuit


# ─────────────────────────────────────────────────────────────
# CIRCUIT SETS
# ─────────────────────────────────────────────────────────────

DEFAULT_CIRCUITS = [
    "data/benchmarks/s820.bench",
    "data/benchmarks/s953.bench",
    "data/benchmarks/s1196.bench",
    "data/benchmarks/s1238.bench",
    "data/benchmarks/c1355.bench",
    "data/benchmarks/s1494.bench",
    "data/benchmarks/s1488.bench",
    "data/benchmarks/c1908.bench",
    "data/benchmarks/c2670.bench",
]

ALL_CIRCUITS = [
    "data/benchmarks/c17.bench",
    "data/benchmarks/s820.bench",
    "data/benchmarks/s832.bench",
    "data/benchmarks/s953.bench",
    "data/benchmarks/c880.bench",
    "data/benchmarks/s1196.bench",
    "data/benchmarks/s1238.bench",
    "data/benchmarks/c1355.bench",
    "data/benchmarks/s1488.bench",
    "data/benchmarks/s1494.bench",
    "data/benchmarks/c1908.bench",
    "data/benchmarks/c2670.bench",
    "data/benchmarks/c3540.bench",
    "data/benchmarks/c5315.bench",
    "data/benchmarks/s5378.bench",
    "data/benchmarks/c7552.bench",
]

N_SEEDS_DEFAULT = 5
MODEL_PATH      = "agent/ppo_checkpoint_curriculum.pt"
RESULTS_PATH    = "agent/benchmark_results.json"


# ─────────────────────────────────────────────────────────────
# SINGLE-RUN METHODS  (each seeded independently)
# ─────────────────────────────────────────────────────────────

def run_sa(circuit, seed: int) -> float:
    random.seed(seed)
    iters = min(200, max(10, circuit.gate_count // 50))
    _, cost, _ = simulated_annealing(
        circuit, initial_temp=100.0, cooling_rate=0.95,
        min_temp=0.1, iterations_per_temp=iters,
        validate=False, verbose=False
    )
    return cost


def run_ppo_stochastic(circuit_path, circuit, agent, predictor, seed, k=8):
    """Samples K rollouts from stochastic policy, keeps best."""
    random.seed(seed)
    torch.manual_seed(seed)
    best_cost  = circuit.cost
    best_gates = circuit.gates

    for _ in range(k):
        env = ChipEnv([circuit_path], predictor, validate=False)
        obs, _ = env.reset()
        for _ in range(env.max_steps):
            action, _, _ = agent.select_action(obs)   # stochastic, not greedy
            obs, _, done, _, _ = env.step(action)
            if done:
                break
        if env.current_cost < best_cost:
            best_cost  = env.current_cost
            best_gates = env.current_gates

    return best_cost, best_gates


def run_ppo_sa_hybrid(circuit_path, circuit, agent: PPOAgent,
                       predictor: GNNPredictor, seed: int) -> float:
    ppo_cost, ppo_gates = run_ppo_stochastic(circuit_path, circuit, agent,
                                    predictor, seed)

    ppo_circuit = Circuit(
        name=circuit.name + "_ppo",
        inputs=circuit.inputs, outputs=circuit.outputs,
        gates=ppo_gates,
        graph=build_graph(circuit.inputs, circuit.outputs, ppo_gates)
    )
    ppo_circuit.cost = ppo_cost
    ppo_circuit = extract_features(ppo_circuit)

    random.seed(seed)
    iters = min(200, max(10, circuit.gate_count // 50))
    _, sa_cost, _ = simulated_annealing(
        ppo_circuit, initial_temp=100.0, cooling_rate=0.95,
        min_temp=0.1, iterations_per_temp=iters,
        validate=False, verbose=False
    )
    return sa_cost, ppo_cost


# ─────────────────────────────────────────────────────────────
# PER-CIRCUIT BENCHMARK
# ─────────────────────────────────────────────────────────────

def benchmark_circuit(path, agent, predictor, n_seeds, verbose=True):
    circuit, _ = load_circuit(path, verbose=False)
    original = circuit.cost

    sa_imps, ppo_imps, hybrid_imps = [], [], []

    for seed in range(n_seeds):
        sa_cost = run_sa(circuit, seed)
        sa_imps.append((original - sa_cost) / original * 100)

        hybrid_cost, ppo_cost = run_ppo_sa_hybrid(
            path, circuit, agent, predictor, seed)
        ppo_imps.append((original - ppo_cost) / original * 100)
        hybrid_imps.append((original - hybrid_cost) / original * 100)
        portfolio_imps = [max(s, h) for s, h in zip(sa_imps, hybrid_imps)]


    result = {
        'circuit': circuit.name, 'gates': circuit.gate_count,
        'original': round(original, 4),
        'sa_mean': round(np.mean(sa_imps), 3),
        'sa_std':  round(np.std(sa_imps), 3),
        'ppo_mean': round(np.mean(ppo_imps), 3),
        'ppo_std':  round(np.std(ppo_imps), 3),
        'hybrid_mean': round(np.mean(hybrid_imps), 3),
        'hybrid_std':  round(np.std(hybrid_imps), 3),
        'n_seeds': n_seeds,
        'portfolio_mean': round(np.mean(portfolio_imps), 3),
        'portfolio_std':  round(np.std(portfolio_imps), 3),
    }

    if verbose:
        print(f"  {circuit.name:<10} {circuit.gate_count:>6}   "
              f"SA {result['sa_mean']:>6.2f}±{result['sa_std']:<5.2f}  "
              f"PPO {result['ppo_mean']:>6.2f}±{result['ppo_std']:<5.2f}  "
              f"Hybrid {result['hybrid_mean']:>6.2f}±{result['hybrid_std']:<5.2f}")

    return result


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--seeds', type=int, default=N_SEEDS_DEFAULT)
    args = parser.parse_args()

    circuits = ALL_CIRCUITS if args.all else DEFAULT_CIRCUITS

    print("=" * 78)
    print("  PPO / SA / PPO+SA HYBRID — STATISTICAL BENCHMARK")
    print("=" * 78)
    print(f"  Circuits : {len(circuits)}")
    print(f"  Seeds    : {args.seeds} per circuit per method")
    print("-" * 78)

    predictor = GNNPredictor()
    agent     = PPOAgent(state_dim=64, n_actions=6, entropy_coef=0.05)
    agent.load(MODEL_PATH)

    print("-" * 78)
    print(f"  {'Circuit':<10} {'Gates':>6}   "
          f"{'SA (mean±std)':<16}  {'PPO (mean±std)':<16}  "
          f"{'Hybrid (mean±std)'}")
    print("-" * 78)

    results = []
    t_start = time.perf_counter()

    for path in circuits:
        if not os.path.exists(path):
            print(f"  [SKIP] {path}")
            continue
        r = benchmark_circuit(path, agent, predictor, args.seeds)
        results.append(r)

    total_time = round(time.perf_counter() - t_start, 1)

    # ── Aggregate summary ──────────────────────────────
    sa_all     = [r['sa_mean']     for r in results]
    ppo_all    = [r['ppo_mean']    for r in results]
    hybrid_all = [r['hybrid_mean'] for r in results]

    wins_hybrid = sum(1 for r in results if r['hybrid_mean'] > r['sa_mean'])

    print("-" * 78)
    print("  AGGREGATE (mean across all circuits)")
    print(f"    SA     : {round(np.mean(sa_all), 3)}%")
    print(f"    PPO    : {round(np.mean(ppo_all), 3)}%")
    print(f"    Hybrid : {round(np.mean(hybrid_all), 3)}%")
    print(f"    Hybrid beats SA on {wins_hybrid}/{len(results)} circuits")
    print(f"    Total runtime: {total_time}s")
    print("=" * 78)

    with open(RESULTS_PATH, 'w') as f:
        json.dump({
            'results': results,
            'aggregate': {
                'sa_mean': round(np.mean(sa_all), 3),
                'ppo_mean': round(np.mean(ppo_all), 3),
                'hybrid_mean': round(np.mean(hybrid_all), 3),
                'hybrid_wins': wins_hybrid,
                'total_circuits': len(results),
            }
        }, f, indent=2)
    print(f"  Saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()