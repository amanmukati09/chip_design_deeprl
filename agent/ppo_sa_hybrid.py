# agent/ppo_sa_hybrid.py
# PPO explores globally (learned circuit-aware policy),
# SA refines locally from PPO's result.
# Mirrors optimizer/hybrid_optimizer.py (GA->SA) architecture.

import sys, os, time, copy
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ml.predictor import GNNPredictor
from environment.chip_env import ChipEnv
from agent.ppo_agent import PPOAgent
from optimizer.simulated_annealing import simulated_annealing
from optimizer.cost_function import compute_pac_cost
from core.circuit import Circuit
from core.graph_builder import build_graph
from core.feature_extractor import extract_features
from core.pipeline import load_circuit


def ppo_sa_hybrid(circuit_path: str, agent: PPOAgent,
                   predictor: GNNPredictor, verbose: bool = True):
    """
    Phase 1: PPO runs its learned policy to explore/mutate circuit.
    Phase 2: SA refines from PPO's result.
    Returns final gates, cost, and improvement report.
    """
    circuit, _ = load_circuit(circuit_path, verbose=False)
    original_cost = circuit.cost

    if verbose:
        print(f"\n  {circuit.name} ({circuit.gate_count} gates)  "
              f"original={round(original_cost,4)}")

    # ── Phase 1: PPO exploration ──────────────────────
    t0 = time.perf_counter()
    env = ChipEnv([circuit_path], predictor, validate=False)
    obs, _ = env.reset()
    for _ in range(env.max_steps):
        action = agent.select_action_greedy(obs)
        obs, _, done, _, _ = env.step(action)
        if done:
            break
    ppo_gates = env.current_gates
    ppo_cost  = env.current_cost
    ppo_time  = time.perf_counter() - t0
    ppo_imp   = (original_cost - ppo_cost) / original_cost * 100

    if verbose:
        print(f"    PPO   : {round(ppo_cost,4)}  "
              f"({round(ppo_imp,2)}%)  {round(ppo_time,2)}s")

    # ── Phase 2: SA refinement from PPO's result ──────
    ppo_circuit = Circuit(
        name=circuit.name + "_ppo",
        inputs=circuit.inputs, outputs=circuit.outputs,
        gates=ppo_gates,
        graph=build_graph(circuit.inputs, circuit.outputs, ppo_gates)
    )
    ppo_circuit.cost = ppo_cost
    ppo_circuit = extract_features(ppo_circuit)

    t0 = time.perf_counter()
    sa_iters = min(200, max(10, circuit.gate_count // 50))
    _, sa_cost, _ = simulated_annealing(
        ppo_circuit, initial_temp=100.0, cooling_rate=0.95,
        min_temp=0.1, iterations_per_temp=sa_iters,
        validate=False, verbose=False
    )
    sa_time = time.perf_counter() - t0
    total_imp = (original_cost - sa_cost) / original_cost * 100

    if verbose:
        print(f"    +SA   : {round(sa_cost,4)}  "
              f"({round(total_imp,2)}% total)  {round(sa_time,2)}s")

    return {
        'circuit': circuit.name, 'gates': circuit.gate_count,
        'original': original_cost,
        'ppo_cost': ppo_cost, 'ppo_imp': round(ppo_imp, 2),
        'final_cost': sa_cost, 'total_imp': round(total_imp, 2),
        'ppo_time': round(ppo_time, 2), 'sa_time': round(sa_time, 2),
    }


if __name__ == "__main__":
    predictor = GNNPredictor()
    agent = PPOAgent(state_dim=64, n_actions=6, entropy_coef=0.05)
    agent.load("agent/ppo_checkpoint_curriculum.pt")

    from optimizer.simulated_annealing import simulated_annealing as sa_baseline

    TEST_CIRCUITS = [
        "data/benchmarks/s1196.bench",
        "data/benchmarks/s1238.bench",
        "data/benchmarks/s1488.bench",
        "data/benchmarks/c1908.bench",
    ]

    print("=" * 70)
    print("  PPO -> SA HYBRID  vs  SA ALONE")
    print("=" * 70)
    print(f"  {'Circuit':<10} {'Gates':>6} {'SA%':>8} {'PPO%':>8} {'PPO+SA%':>9}")
    print("-" * 70)

    for path in TEST_CIRCUITS:
        circuit, _ = load_circuit(path, verbose=False)
        _, sa_cost, _ = sa_baseline(
            circuit, initial_temp=100.0, cooling_rate=0.95,
            min_temp=0.1, iterations_per_temp=10,
            validate=False, verbose=False)
        sa_imp = (circuit.cost - sa_cost) / circuit.cost * 100

        result = ppo_sa_hybrid(path, agent, predictor, verbose=False)

        print(f"  {result['circuit']:<10} {result['gates']:>6} "
              f"{sa_imp:>7.2f}% {result['ppo_imp']:>7.2f}% "
              f"{result['total_imp']:>8.2f}%")

    print("=" * 70)