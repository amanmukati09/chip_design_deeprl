# environment/chip_env.py
# Gymnasium environment wrapping circuit + mutation engine for PPO.
#
# MDP design:
#   State  : GNN embedding of current circuit (float vector, fixed size)
#   Action : which mutation rule to apply (0-5, discrete)
#   Reward : PAC cost reduction (positive = good)
#   Episode: max_steps mutations on one circuit
#   Done   : max_steps reached

import gymnasium as gym
import numpy as np
import copy
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gymnasium import spaces
from optimizer.cost_function import compute_pac_cost
from optimizer.mutations     import apply_safe_mutation, MUTATION_RULES
from ml.predictor            import GNNPredictor
from ml.data_collector       import gates_to_graph_data
import torch

# Fixed embedding size — GNN mean+max pool → hidden_dim*2 = 64
EMBEDDING_DIM = 64
MAX_STEPS     = 50   # mutations per episode


class ChipEnv(gym.Env):
    """
    Circuit optimization environment for PPO.

    One episode = optimizing one circuit for MAX_STEPS steps.
    Agent picks which mutation rule to apply each step.
    Reward = normalized PAC cost reduction.
    """

    metadata = {"render_modes": []}

    def __init__(self, circuit_paths: list,
                 predictor: GNNPredictor = None,
                 max_steps: int = MAX_STEPS,
                 validate: bool = False):
        super().__init__()

        self.circuit_paths = circuit_paths
        self.predictor     = predictor
        self.max_steps     = max_steps
        self.validate      = validate

        # 6 mutation rules (matching MUTATION_RULES in mutations.py)
        self.action_space      = spaces.Discrete(6)

        # State = GNN embedding (mean + max pooled → 64-dim)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(EMBEDDING_DIM,), dtype=np.float32
        )

        # Set at reset()
        self.circuit       = None
        self.current_gates = None
        self.original_cost = None
        self.current_cost  = None
        self.step_count    = 0

    # ──────────────────────────────────────────────
    def _get_embedding(self, gates) -> np.ndarray:
        """Runs GNN on current gates, returns embedding vector."""
        node_feat, edge_index = gates_to_graph_data(
            gates,
            self.circuit.inputs,
            self.circuit.outputs,
            self.circuit.gate_count
        )
        feat_tensor = torch.tensor(node_feat, dtype=torch.float32)

        with torch.no_grad():
            x = feat_tensor
            num_nodes = x.size(0)

            # Two SAGE conv layers (matches SmallCircuitGNN in trainer.py)
            model = self.predictor.model
            x1    = model.conv1(x, edge_index, num_nodes)
            x1    = model.drop(x1)
            x2    = model.conv2(x1, edge_index, num_nodes)
            x     = x1 + x2   # residual

            # Mean + max pooling → 64-dim embedding
            emb_mean = x.mean(dim=0)
            emb_max  = x.max(dim=0)[0]
            embedding = torch.cat([emb_mean, emb_max], dim=0)

        return embedding.numpy().astype(np.float32)

    # ──────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Pick a random circuit each episode
        import random
        from core.pipeline import load_circuit
        path           = random.choice(self.circuit_paths)
        self.circuit, _ = load_circuit(path, verbose=False)

        self.current_gates = copy.deepcopy(self.circuit.gates)
        self.original_cost = self.circuit.cost
        self.current_cost  = self.circuit.cost
        self.step_count    = 0

        obs = self._get_embedding(self.current_gates)
        return obs, {}

    # ──────────────────────────────────────────────
    def step(self, action: int):
        self.step_count += 1

        # Apply the chosen mutation rule
        new_gates = apply_safe_mutation(
            self.circuit.inputs,
            self.circuit.outputs,
            self.current_gates,
            rule_index  = action,
            max_attempts = 10,
            validate    = self.validate
        )

        if new_gates is None:
            # Mutation failed — small penalty, stay in place
            reward    = -0.01
            obs       = self._get_embedding(self.current_gates)
            done      = self.step_count >= self.max_steps
            return obs, reward, done, False, {}

        new_cost = compute_pac_cost(new_gates, self.circuit.inputs)['total_cost']

        # Reward = cost reduction normalized by original cost
        reward             = (self.current_cost - new_cost) / self.original_cost
        self.current_gates = new_gates
        self.current_cost  = new_cost

        obs  = self._get_embedding(self.current_gates)
        done = self.step_count >= self.max_steps

        info = {
            'current_cost'  : round(new_cost, 4),
            'improvement_pct': round(
                (self.original_cost - new_cost) / self.original_cost * 100, 3)
        }
        return obs, reward, done, False, info

    # ──────────────────────────────────────────────
    def render(self):
        print(f"  Step {self.step_count}/{self.max_steps}  "
              f"cost={round(self.current_cost,4)}  "
              f"improvement="
              f"{round((self.original_cost-self.current_cost)/self.original_cost*100,2)}%")