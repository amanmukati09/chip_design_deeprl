# agent/test_k_scaling.py
# Tests whether increasing K (number of stochastic PPO rollouts sampled)
# closes the gap on c1908 and c2670, where best-of-8 hybrid underperformed SA.
#
# Hypothesis: larger circuits have a bigger action-sequence search space,
# so K=8 samples may be insufficient to find a good PPO trajectory that
# seeds SA into a competitive basin.

import sys, os, time, random
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np

from ml.predictor                  import GNNPredictor
from environment.chip_env          import ChipEnv
from agent.ppo_agent                import PPOAgent
from optimizer.simulated_annealing import simulated_annealing
from core.circuit                  import Circuit
from core.graph_builder            import build_graph
from core.feature_extractor        import extract_features
from core.pipeline                 import load_circuit

MODEL_PATH = "agent/ppo_checkpoint_curriculum.pt"
TEST_CIRCUITS = [
    "data/benchmarks/c1908.bench",
    "data/benchmarks/c2670.bench",
]
K_VALUES = [8, 16, 32]
N_SEEDS  = 3   # fewer seeds per K since we're testing 3 K values x 2 circuits


def run_ppo_stochastic(circuit_path, circuit, agent, predictor, seed, k):
    random.seed(seed)
    torch.manual_seed(seed)
    best_cost  = circuit.cost
    best_gates = circuit.gates

    for _ in range(k):
        env = ChipEnv([circuit_path], predictor, validate=False)
        obs, _ = env.reset()
        for _ in range(env.max_steps):
            action, _, _ = agent.select_action(obs)
            obs, _, done, _, _ = env.step(action)
            if done:
                break
        if env.current_cost < best_cost:
            best_cost  = env.current_cost
            best_gates = env.current_gates

    return best_cost, best_gates


def run_hybrid(circuit_path, circuit, agent, predictor, seed, k):
    ppo_cost, ppo_gates = run_ppo_stochastic(
        circuit_path, circuit, agent, predictor, seed, k)

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


def run_sa_baseline(circuit, seed):
    random.seed(seed)
    iters = min(200, max(10, circuit.gate_count // 50))
    _, cost, _ = simulated_annealing(
        circuit, initial_temp=100.0, cooling_rate=0.95,
        min_temp=0.1, iterations_per_temp=iters,
        validate=False, verbose=False
    )
    return cost


def main():
    print("=" * 74)
    print("  K-SCALING TEST — c1908 & c2670 (hybrid underperformed SA at K=8)")
    print("=" * 74)

    predictor = GNNPredictor()
    agent     = PPOAgent(state_dim=64, n_actions=6, entropy_coef=0.05)
    agent.load(MODEL_PATH)

    print("-" * 74)
    print(f"  {'Circuit':<10} {'K':>4}   {'SA':>8}   "
          f"{'PPO(best-of-K)':>15}   {'Hybrid':>15}")
    print("-" * 74)

    for path in TEST_CIRCUITS:
        circuit, _ = load_circuit(path, verbose=False)
        original   = circuit.cost

        sa_imps = [run_sa_baseline(circuit, s) for s in range(N_SEEDS)]
        sa_mean = np.mean([(original - c) / original * 100 for c in sa_imps])

        for k in K_VALUES:
            t0 = time.perf_counter()
            ppo_imps, hyb_imps = [], []

            for seed in range(N_SEEDS):
                hyb_cost, ppo_cost = run_hybrid(
                    path, circuit, agent, predictor, seed, k)
                ppo_imps.append((original - ppo_cost) / original * 100)
                hyb_imps.append((original - hyb_cost) / original * 100)

            elapsed = round(time.perf_counter() - t0, 1)
            ppo_mean = round(np.mean(ppo_imps), 2)
            ppo_std  = round(np.std(ppo_imps), 2)
            hyb_mean = round(np.mean(hyb_imps), 2)
            hyb_std  = round(np.std(hyb_imps), 2)

            marker = " <-- beats SA" if hyb_mean > sa_mean else ""

            print(f"  {circuit.name:<10} {k:>4}   {round(sa_mean,2):>7.2f}%  "
                  f"{ppo_mean:>7.2f}±{ppo_std:<5.2f}  "
                  f"{hyb_mean:>7.2f}±{hyb_std:<5.2f}  "
                  f"({elapsed}s){marker}")

        print("-" * 74)

    print("=" * 74)


if __name__ == "__main__":
    main()