# agent/evaluate_ppo.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ml.predictor         import GNNPredictor
from environment.chip_env import ChipEnv
from agent.ppo_agent      import PPOAgent
from optimizer.simulated_annealing import simulated_annealing
from core.pipeline        import load_circuit
import numpy as np

TEST_CIRCUITS = [
    "data/benchmarks/s1196.bench",   # trained on this
    "data/benchmarks/s1238.bench",   # similar size, unseen
    "data/benchmarks/s1488.bench",   # larger, unseen
    "data/benchmarks/c1908.bench",   # different family, unseen
]

predictor = GNNPredictor()
agent     = PPOAgent()
agent.load("agent/ppo_checkpoint_curriculum.pt")

print(f"\n  {'Circuit':<12} {'Gates':>6} {'SA%':>8} {'PPO%':>8}")
print(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*8}")
# Replace the PPO evaluation loop with:

N_RUNS = 5
ppo_results = []

for path in TEST_CIRCUITS:
    circuit, _ = load_circuit(path, verbose=False)

    _, sa_cost, _ = simulated_annealing(
        circuit, initial_temp=100.0, cooling_rate=0.95,
        min_temp=0.1, iterations_per_temp=10,
        validate=False, verbose=False)
    sa_imp = (circuit.cost - sa_cost) / circuit.cost * 100

    run_improvements = []
    for _ in range(N_RUNS):
        env = ChipEnv([path], predictor, validate=False)
        obs, _ = env.reset()
        for _ in range(env.max_steps):
            action = agent.select_action_greedy(obs)
            obs, _, done, _, _ = env.step(action)
            if done:
                break
        imp = (circuit.cost - env.current_cost) / circuit.cost * 100
        run_improvements.append(imp)

    best_ppo = max(run_improvements)
    avg_ppo  = sum(run_improvements) / len(run_improvements)

    print(f"  {circuit.name:<12} {circuit.gate_count:>6} "
          f"{sa_imp:>7.2f}%  best={best_ppo:>6.2f}%  avg={avg_ppo:>6.2f}%")