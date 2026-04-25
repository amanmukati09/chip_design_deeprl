# AI-Driven Circuit Design Optimization System

Research implementation of **"Revolutionizing VLSI: AI-Driven Strategies for Power, Area, and Cost Optimization in Chip Design"**

> Aman Mukati — IIIT Pune, 2024  
> GitHub: [amanmukati09/chip_design_deeprl](https://github.com/amanmukati09/chip_design_deeprl)

---

## What This Does

Takes a `.bench` VLSI circuit file as input and returns an optimized circuit with the best Power-Area-Cost (PAC) balance, using a combination of classical heuristics and graph neural networks.

```
.bench file → Parser → Graph → GNN Cost Predictor → Optimizer → Optimized Circuit
```

---

## Key Results

Benchmarked across 15 MCNC/ITC circuits. GNN-accelerated SA wins on 11/15 circuits.

| Circuit | Gates | SA | GA | GA+SA | GNN-SA | Best |
|---------|-------|-----|-----|-------|--------|------|
| c880    | 383   | 7.7% | 2.76% | 8.38% | 0.0% | **8.38%** |
| s820    | 294   | 5.57% | 2.89% | 5.25% | 0.72% | **5.57%** |
| s832    | 292   | 3.25% | 2.95% | 7.07% | 0.0% | **7.07%** |
| s953    | 424   | 3.93% | 1.49% | 2.76% | 0.0% | **3.93%** |
| s1196   | 547   | 6.12% | 1.65% | 7.42% | **9.48%** | **9.48%** |
| s1238   | 526   | 3.33% | 2.02% | 3.76% | **10.72%** | **10.72%** |
| s1488   | 659   | 6.02% | 1.72% | 6.70% | **14.81%** | **14.81%** |
| s1494   | 653   | 5.52% | 1.81% | 6.01% | **12.26%** | **12.26%** |
| c2670   | 1193  | 10.21% | 0.91% | 9.76% | **21.75%** | **21.75%** |
| c3540   | 1669  | 5.03% | 0.72% | 7.83% | **16.91%** | **16.91%** |
| s5378   | 2574  | 3.24% | 0.26% | 3.13% | **10.19%** | **10.19%** |
| s13207  | 2107  | 3.44% | 0.30% | 3.38% | **9.96%** | **9.96%** |
| c7552   | 3512  | 3.93% | 0.22% | 3.41% | **12.30%** | **12.30%** |
| s35932  | 4535  | 1.32% | 0.06% | 1.34% | **3.95%** | **3.95%** |
| s38584  | 20679 | 0.36% | 0.05% | 0.44% | **0.88%** | **0.88%** |

**Key finding:** GNN-SA advantage grows with circuit complexity — up to **21.75% improvement** on c2670 (1193 gates), consistently outperforming all other methods on circuits ≥ 500 gates.

---

## PAC Cost Formula

```
PAC = 0.4 × Power + 0.4 × Area + 0.2 × WireLength
```

| Gate | Power | Area |
|------|-------|------|
| NOT, BUFF | 1.0 | 1.0 |
| NAND, NOR | 2.0 | 1.2 |
| AND, OR   | 2.5 | 1.5 |
| XOR, XNOR | 4.0 | 2.5 |
| DFF       | 6.0 | 4.0 |

---

## Architecture

```
core/
  netlist_parser.py       Robust .bench parser (handles 20,000+ gates)
  graph_builder.py        NetworkX DiGraph construction
  circuit.py              Circuit object with copy()
  feature_extractor.py    Fan-in, fan-out, depth, gate type features
  pipeline.py             End-to-end connector

optimizer/
  cost_function.py        PAC cost computation
  simulated_annealing.py  SA optimizer
  genetic_algorithm.py    GA optimizer
  hybrid_optimizer.py     GA → SA hybrid
  gnn_optimizer.py        GNN-accelerated SA (novel)
  benchmark.py            Multi-circuit comparison runner

ml/
  data_collector.py       Training data generation (3-tier strategy)
  gnn_model.py            GraphSAGE GNN (manual impl, no PyG needed)
  trainer.py              GNN training loop
  predictor.py            Fast cost prediction (2.5ms per circuit)

heuristics/
  manager.py              Complexity router (the novel contribution)

api/
  main.py                 FastAPI REST endpoint
```

---

## Complexity Router

The system automatically selects the best optimizer based on circuit size — determined empirically from benchmark results:

```python
if gate_count < 500:
    use GA+SA hybrid      # GNN generalizes poorly to small circuits
else:
    use GNN-accelerated SA  # wins on 11/15 circuits ≥ 500 gates
```

Iteration scaling ensures large circuits get adequate exploration:
```python
iterations_per_temp = max(10, gate_count // 50)
```

---

## Setup

```bash
# Clone
git clone https://github.com/amanmukati09/chip_design_deeprl
cd chip_design_deeprl

# Install
pip install torch networkx fastapi uvicorn python-multipart

# Train GNN (optional — pretrained model included)
python ml/data_collector.py
python ml/trainer.py

# Run benchmark
python optimizer/benchmark.py s1196 c2670 s1488

# Start API
python api/main.py
# Open http://localhost:8000/docs
```

---

## API

```bash
# Health check
GET /health

# Optimize a built-in benchmark
GET /optimize/benchmark?name=s1196

# Upload your own .bench file
POST /optimize
  body: multipart/form-data, file=your_circuit.bench

# Force a specific optimizer
GET /optimize/benchmark?name=c2670&optimizer=gnn_sa
```

**Example response:**
```json
{
  "status": "success",
  "circuit": {
    "name": "c2670",
    "gate_count": 1193,
    "original_cost": 1630.286
  },
  "optimization": {
    "optimizer": "gnn_sa",
    "optimized_cost": 1275.6,
    "improvement_pct": 21.75,
    "time_seconds": 12.97,
    "gnn_available": true
  }
}
```

---

## GNN Architecture

Manual GraphSAGE implementation — no PyTorch Geometric required.

```
Input: node features [gate_type, fan_in, fan_out]
  ↓
SAGEConv(3 → 64)   + BatchNorm + ReLU
  ↓
SAGEConv(64 → 128) + BatchNorm + ReLU
  ↓
SAGEConv(128 → 64) + BatchNorm + ReLU
  ↓
Mean Pooling (circuit-level embedding)
  ↓
Linear(64 → 32) → Linear(32 → 1)
  ↓
Predicted PAC Cost
```

- **36,033 parameters** — deliberately lean for CPU training
- **0.40% average prediction error** on validation set
- **2.5ms per prediction** — fast enough for use inside SA loop
- Trained on 2000 samples with 3-tier cost range coverage

---

## Tech Stack

- Python 3.14
- PyTorch (CPU)
- NetworkX
- FastAPI + Uvicorn
- MCNC/ITC benchmark circuits

---

## Project Status

| Phase | Status | Description |
|-------|--------|-------------|
| 1 — Foundation | ✅ Done | Parser, graph, circuit, features |
| 2 — Optimizers | ✅ Done | SA, GA, GA+SA hybrid |
| 3 — GNN | ✅ Done | GraphSAGE, trainer, predictor |
| 4 — API | ✅ Done | FastAPI REST endpoint |
| 5 — Complexity Router | ✅ Done | Auto optimizer selection |
| 6 — PPO Agent | 🔄 Planned | DeepRL policy for mutation selection |

---

## Citation

If you use this code, please cite:

```
Mukati, A. (2024). Revolutionizing VLSI: AI-Driven Strategies for Power, 
Area, and Cost Optimization in Chip Design. IIIT Pune.
```