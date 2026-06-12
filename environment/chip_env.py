# environment/chip_env.py
# Gymnasium MDP environment for PPO circuit optimization.
#
# State  : 64-dim GNN embedding of current circuit
#          (mean + max pool from SmallCircuitGNN, hidden_dim=32 → 32+32=64)
# Action : 0-5 discrete (which of 6 mutation rules to apply)
# Reward : normalized PAC cost reduction this step
# Episode: MAX_STEPS mutation steps on one circuit
# Done   : MAX_STEPS reached

import gymnasium as gym
import numpy as np
import copy
import random
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gymnasium           import spaces
from optimizer.cost_function import compute_pac_cost
from optimizer.mutations     import apply_safe_mutation
from ml.data_collector       import gates_to_graph_data
import torch

EMBEDDING_DIM = 64   # hidden_dim=32, mean+max → 32+32
MAX_STEPS     = 50


class ChipEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, circuit_paths: list,
                 predictor,
                 max_steps: int = MAX_STEPS,
                 validate: bool = False):
        super().__init__()

        self.circuit_paths = circuit_paths
        self.predictor     = predictor
        self.max_steps     = max_steps
        self.validate      = validate

        self.action_space      = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(EMBEDDING_DIM,), dtype=np.float32
        )

        self.circuit       = None
        self.current_gates = None
        self.original_cost = None
        self.current_cost  = None
        self.step_count    = 0

    def _get_embedding(self, gates) -> np.ndarray:
        node_feat, edge_index = gates_to_graph_data(
            gates,
            self.circuit.inputs,
            self.circuit.outputs,
            self.circuit.gate_count
        )
        x = torch.tensor(node_feat, dtype=torch.float32)
        num_nodes = x.size(0)
        model = self.predictor.model

        with torch.no_grad():
            x1 = model.conv1(x, edge_index, num_nodes)
            x1 = model.drop(x1)
            x2 = model.conv2(x1, edge_index, num_nodes)
            x  = x1 + x2   # residual — matches SmallCircuitGNN
            emb = torch.cat([x.mean(dim=0), x.max(dim=0)[0]], dim=0)

        return emb.numpy().astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        from core.pipeline import load_circuit

        path            = random.choice(self.circuit_paths)
        self.circuit, _ = load_circuit(path, verbose=False)

        self.current_gates = copy.deepcopy(self.circuit.gates)
        self.original_cost = self.circuit.cost
        self.current_cost  = self.circuit.cost
        self.step_count    = 0

        return self._get_embedding(self.current_gates), {}
    
    def step(self, action: int):
        
        self.step_count += 1

        new_gates = apply_safe_mutation(
            self.circuit.inputs, self.circuit.outputs,
            self.current_gates,
            rule_index=action, max_attempts=10,
            validate=self.validate
        )

        done = self.step_count >= self.max_steps

        if new_gates is None:
            obs = self._get_embedding(self.current_gates)
            # Episode-end reward only
            reward = ((self.original_cost - self.current_cost)
                    / self.original_cost) if done else 0.0
            return obs, reward, done, False, {}

        new_cost           = compute_pac_cost(new_gates, self.circuit.inputs)['total_cost']
        self.current_gates = new_gates
        self.current_cost  = new_cost

        obs = self._get_embedding(self.current_gates)

        # Reward only at episode end
        reward = ((self.original_cost - self.current_cost)
                / self.original_cost) if done else 0.0

        info = {
            'current_cost'   : round(self.current_cost, 4),
            'improvement_pct': round(
                (self.original_cost - self.current_cost)
                / self.original_cost * 100, 3)
        }
        return obs, reward, done, False, info



    def render(self):
        print(f"  Step {self.step_count}/{self.max_steps}  "
              f"cost={round(self.current_cost,4)}  "
              f"improvement="
              f"{round((self.original_cost-self.current_cost)/self.original_cost*100,2)}%")