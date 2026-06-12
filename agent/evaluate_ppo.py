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
agent.load("agent/ppo_checkpoint.pt")

print(f"\n  {'Circuit':<12} {'Gates':>6} {'SA%':>8} {'PPO%':>8}")
print(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*8}")

for path in TEST_CIRCUITS:
    circuit, _ = load_circuit(path, verbose=False)

    # SA baseline
    _, sa_cost, _ = simulated_annealing(
        circuit, initial_temp=100.0, cooling_rate=0.95,
        min_temp=0.1, iterations_per_temp=10,
        validate=False, verbose=False)
    sa_imp = (circuit.cost - sa_cost) / circuit.cost * 100

    # PPO
    env = ChipEnv([path], predictor, max_steps=50, validate=False)
    obs, _ = env.reset()
    for _ in range(50):
        action, _, _ = agent.select_action(obs)
        obs, _, done, _, _ = env.step(action)
        if done:
            break
    ppo_imp = (circuit.cost - env.current_cost) / circuit.cost * 100

    print(f"  {circuit.name:<12} {circuit.gate_count:>6} "
          f"{sa_imp:>7.2f}% {ppo_imp:>7.2f}%")