# agent/train_ppo.py
# PPO training loop for circuit optimization.
# Run from project root: python agent/train_ppo.py

import sys
import os
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ml.predictor          import GNNPredictor
from environment.chip_env  import ChipEnv
from agent.ppo_agent       import PPOAgent

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

TRAIN_CIRCUITS = [
    # "data/benchmarks/s820.bench",
    # "data/benchmarks/s953.bench",
    "data/benchmarks/s1196.bench",
    # "data/benchmarks/s1238.bench",
    # "data/benchmarks/c1355.bench",
    # "data/benchmarks/s1494.bench",
    # "data/benchmarks/c1908.v",
    # "data/benchmarks/c2670.v",
    # "data/benchmarks/c6288.v",
    # "data/benchmarks/c7552.v",
    # "data/benchmarks/s9234.bench",
    # "data/benchmarks/s1494.bench",
    # "data/benchmarks/s5378.bench",
    # "data/benchmarks/hyp.bench",

]

N_EPISODES    = 2000
MAX_STEPS     = 50
SAVE_EVERY    = 50
MODEL_PATH    = "agent/ppo_checkpoint.pt"
LOG_PATH      = "agent/ppo_log.json"


def train():
    print("=" * 55)
    print("  PPO Training — Circuit Optimization")
    print("=" * 55)

    predictor = GNNPredictor()
    env       = ChipEnv(TRAIN_CIRCUITS, predictor,
                         max_steps=MAX_STEPS, validate=False)
    
    
    agent = PPOAgent(state_dim=64, n_actions=6, entropy_coef=0.05)

    print(f"  Circuits  : {len(TRAIN_CIRCUITS)}")
    print(f"  Episodes  : {N_EPISODES}")
    print(f"  Max steps : {MAX_STEPS}")
    print(f"  Save every: {SAVE_EVERY} episodes")
    print("-" * 55)
    print(f"  {'Ep':>5}  {'Reward':>8}  {'Improve%':>9}  "
          f"{'Loss':>8}  {'Steps':>6}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*6}")

    log = []

    for episode in range(1, N_EPISODES + 1):
        obs, _        = env.reset()
        ep_reward     = 0.0
        ep_improve    = 0.0
        steps_taken   = 0

        for step in range(MAX_STEPS):
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, done, _, info = env.step(action)

            agent.buffer.store(
                obs, action, reward, done, log_prob, value)

            obs         = next_obs
            ep_reward  += reward
            steps_taken = step + 1

            if done:
                ep_improve = info.get('improvement_pct', 0.0)
                break

        # Update after each episode
        loss = agent.update()

        log.append({
            'episode'    : episode,
            'reward'     : round(ep_reward, 5),
            'improvement': round(ep_improve, 3),
            'loss'       : loss,
            'steps'      : steps_taken,
        })

        if episode % 10 == 0 or episode == 1:
            print(f"  {episode:>5}  {ep_reward:>8.4f}  "
                  f"{ep_improve:>8.3f}%  {loss:>8.5f}  "
                  f"{steps_taken:>6}")

        if episode % SAVE_EVERY == 0:
            agent.save(MODEL_PATH)
            with open(LOG_PATH, 'w') as f:
                json.dump(log, f, indent=2)

    # Final save
    agent.save(MODEL_PATH)
    with open(LOG_PATH, 'w') as f:
        json.dump(log, f, indent=2)

    print("-" * 55)
    print(f"  Training complete.")
    print(f"  Model : {MODEL_PATH}")
    print(f"  Log   : {LOG_PATH}")
    print("=" * 55)


if __name__ == "__main__":
    train()