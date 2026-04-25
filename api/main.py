"""
api/main.py
─────────────────────────────────────────────────────
FastAPI endpoint for AI-Driven Circuit Design Optimization.

Endpoints:
    POST /optimize          → upload .bench file, get optimization report
    GET  /health            → service status
    GET  /benchmarks        → list available built-in benchmarks
    POST /optimize/benchmark → optimize a built-in benchmark by name

Usage:
    pip install fastapi uvicorn python-multipart
    python api/main.py

Then open: http://localhost:8000/docs
"""

import os
import sys
import time
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

from core.pipeline                 import load_circuit
from optimizer.simulated_annealing import simulated_annealing
from optimizer.genetic_algorithm   import genetic_algorithm
from optimizer.hybrid_optimizer    import hybrid_optimize
from optimizer.gnn_optimizer       import gnn_simulated_annealing
from heuristics.manager import optimize as manager_optimize, scale_iterations



# ─────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "AI-Driven Circuit Design Optimization API",
    description = (
        "Optimizes VLSI circuits from .bench files using SA, GA, "
        "GA+SA hybrid, and GNN-accelerated SA. "
        "Returns the best Power-Area-Cost (PAC) result."
    ),
    version = "1.0.0",
)

BENCHMARK_DIR = "data/benchmarks"

# ── Load GNN predictor once at startup ───────────────────────
try:
    from ml.predictor import GNNPredictor
    predictor = GNNPredictor()
    GNN_AVAILABLE = True
    print("[API] GNN predictor loaded.")
except Exception as e:
    predictor     = None
    GNN_AVAILABLE = False
    print(f"[API] GNN predictor unavailable: {e}")


# ─────────────────────────────────────────────────────────────
# OPTIMIZER SELECTION
# GNN-SA wins on circuits ≥500 gates
# GA+SA wins on smaller circuits
# ─────────────────────────────────────────────────────────────

def select_optimizer(gate_count: int, override: Optional[str] = None) -> str:
    if override and override in ("sa", "ga", "hybrid", "gnn_sa"):
        return override
    if not GNN_AVAILABLE:
        return "hybrid"
    return "gnn_sa" if gate_count >= 500 else "hybrid"


def run_optimizer(circuit, optimizer: str):
    result = manager_optimize(circuit, force_optimizer=optimizer, verbose=False)
    return result["optimized_cost"], result["detail"], result["time_seconds"]


# ─────────────────────────────────────────────────────────────
# RESPONSE BUILDER
# ─────────────────────────────────────────────────────────────

def build_response(circuit, cost, report, optimizer, elapsed):
    original     = circuit.cost
    improvement  = round((original - cost) / original * 100, 4)

    return {
        "status"      : "success",
        "circuit"     : {
            "name"          : circuit.name,
            "gate_count"    : circuit.gate_count,
            "original_cost" : round(original, 4),
        },
        "optimization": {
            "optimizer"     : optimizer,
            "optimized_cost": round(cost, 4),
            "improvement_pct": improvement,
            "time_seconds"  : elapsed,
            "gnn_available" : GNN_AVAILABLE,
        },
        "detail"      : report,
    }


# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Check if the API and GNN predictor are ready."""
    return {
        "status"       : "ok",
        "gnn_available": GNN_AVAILABLE,
        "benchmarks_dir": os.path.abspath(BENCHMARK_DIR),
    }


@app.get("/benchmarks")
def list_benchmarks():
    """List all available built-in benchmark circuits."""
    if not os.path.exists(BENCHMARK_DIR):
        return {"benchmarks": []}

    files = [
        f.replace(".bench", "")
        for f in os.listdir(BENCHMARK_DIR)
        if f.endswith(".bench") and f != "test.bench"
    ]
    files.sort()
    return {"benchmarks": files, "count": len(files)}


@app.post("/optimize")
async def optimize_upload(
    file     : UploadFile = File(..., description="Upload a .bench circuit file"),
    optimizer: Optional[str] = Query(
        default=None,
        description="Force optimizer: sa | ga | hybrid | gnn_sa. "
                    "Auto-selected if not specified."
    )
):
    """
    Upload a .bench file and get an optimized circuit report.

    Auto-selects the best optimizer based on circuit size:
    - GNN-SA for circuits ≥500 gates
    - GA+SA hybrid for smaller circuits
    """
    # Validate file type
    if not file.filename.endswith(".bench"):
        raise HTTPException(
            status_code=400,
            detail="Only .bench files are supported."
        )

    # Save upload to temp file
    content = await file.read()
    with tempfile.NamedTemporaryFile(
        suffix=".bench", delete=False, mode="wb"
    ) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Load circuit
        circuit, _ = load_circuit(tmp_path)

        # Select and run optimizer
        opt = select_optimizer(circuit.gate_count, optimizer)
        cost, report, elapsed = run_optimizer(circuit, opt)

        return build_response(circuit, cost, report, opt, elapsed)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        os.unlink(tmp_path)


@app.post("/optimize/benchmark")
def optimize_benchmark(
    name     : str = Query(..., description="Benchmark name e.g. s1196, c2670"),
    optimizer: Optional[str] = Query(
        default=None,
        description="Force optimizer: sa | ga | hybrid | gnn_sa"
    )
):
    """
    Optimize a built-in benchmark circuit by name.
    No file upload needed — just pass the circuit name.
    """
    filepath = os.path.join(BENCHMARK_DIR, f"{name}.bench")
    if not os.path.exists(filepath):
        raise HTTPException(
            status_code=404,
            detail=f"Benchmark '{name}' not found. "
                   f"Call GET /benchmarks to see available circuits."
        )

    try:
        circuit, _ = load_circuit(filepath)
        opt = select_optimizer(circuit.gate_count, optimizer)
        cost, report, elapsed = run_optimizer(circuit, opt)
        return build_response(circuit, cost, report, opt, elapsed)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/optimize/benchmark")
def optimize_benchmark_get(
    name     : str = Query(..., description="Benchmark name e.g. s1196"),
    optimizer: Optional[str] = Query(default=None)
):
    """Same as POST /optimize/benchmark but via GET for easy browser testing."""
    return optimize_benchmark(name=name, optimizer=optimizer)


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Circuit Optimization API")
    print("  http://localhost:8000")
    print("  http://localhost:8000/docs  ← interactive UI")
    print("=" * 55)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)