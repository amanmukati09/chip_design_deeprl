# agent/train_ppo_curriculum.py
# Curriculum PPO: train on easy circuits first, progressively add harder ones.
# Same hyperparameters as combo 4 (entropy=0.05, episode-end reward)
# which was our stable baseline.

import sys, os, json
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ml.predictor          import GNNPredictor
from environment.chip_env  import ChipEnv
from agent.ppo_agent       import PPOAgent

MAX_STEPS  = 50
MODEL_PATH = "agent/ppo_checkpoint_curriculum.pt"
LOG_PATH   = "agent/ppo_log_curriculum.json"

# Curriculum stages — circuits added progressively   
STAGES = [
    {"episodes": 400, "circuits": ["data/benchmarks/s820.bench"]},
    {"episodes": 400, "circuits": ["data/benchmarks/s820.bench",
                                    "data/benchmarks/s953.bench"]},
    {"episodes": 400, "circuits": ["data/benchmarks/s820.bench",
                                    "data/benchmarks/s953.bench",
                                    "data/benchmarks/s1196.bench"]},
    {"episodes": 400, "circuits": ["data/benchmarks/s820.bench",
                                    "data/benchmarks/s953.bench",
                                    "data/benchmarks/s1196.bench",
                                    "data/benchmarks/s1238.bench",
                                    "data/benchmarks/c1355.bench",
                                    "data/benchmarks/s1494.bench"]},
    {"episodes": 400, "circuits": ["data/benchmarks/s820.bench",
                                    "data/benchmarks/s953.bench",
                                    "data/benchmarks/s1196.bench",
                                    "data/benchmarks/s1238.bench",
                                    "data/benchmarks/c1355.bench",
                                    "data/benchmarks/s1494.bench",
                                    "data/benchmarks/c1908.bench",
                                    "data/benchmarks/s1488.bench"]},
]


def train():
    print("=" * 60)
    print("  CURRICULUM PPO TRAINING")
    print("=" * 60)

    predictor = GNNPredictor()
    agent     = PPOAgent(state_dim=64, n_actions=6, entropy_coef=0.05)

    log = []
    global_ep = 0

    for stage_num, stage in enumerate(STAGES, 1):
        circuits = stage["circuits"]
        episodes = stage["episodes"]
        env = ChipEnv(circuits, predictor, validate=False)
        
        print(f"\n  STAGE {stage_num}: {len(circuits)} circuit(s), "
              f"{episodes} episodes")
        print(f"  Circuits: {[os.path.basename(c) for c in circuits]}")
        print(f"  {'Ep':>6}  {'Reward':>8}  {'Improve%':>9}  {'Loss':>8}")

        for ep in range(1, episodes + 1):
            global_ep += 1
            obs, _     = env.reset()
            ep_reward  = 0.0
            ep_improve = 0.0

            for step in range(env.max_steps):
                action, log_prob, value = agent.select_action(obs)
                next_obs, reward, done, _, info = env.step(action)
                agent.buffer.store(obs, action, reward, done,
                                    log_prob, value)
                obs        = next_obs
                ep_reward += reward
                if done:
                    ep_improve = info.get('improvement_pct', 0.0)
                    break

            loss = agent.update()
            log.append({
                'global_episode': global_ep, 'stage': stage_num,
                'reward': round(ep_reward, 5),
                'improvement': round(ep_improve, 3), 'loss': loss
            })

            if ep % 50 == 0 or ep == 1:
                print(f"  {ep:>6}  {ep_reward:>8.4f}  "
                      f"{ep_improve:>8.3f}%  {loss:>8.5f}")

        agent.save(MODEL_PATH)
        with open(LOG_PATH, 'w') as f:
            json.dump(log, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  Curriculum training complete. {global_ep} total episodes.")
    print(f"  Model: {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    train()