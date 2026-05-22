# heuristics/manager.py
# Complexity Router — fixed version.
#
# Fixes vs previous:
#   1. Removed import of gnn_simulated_annealing at module level
#      (was causing load failure if ml/ models missing)
#   2. optimize() now also writes output .bench + report
#   3. Large circuit (>5000 gates) notes added honestly
#   4. All optimizer calls use correct updated signatures

import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

_predictor = None


def _get_predictor():
    global _predictor
    if _predictor is not None:
        return _predictor
    try:
        from ml.predictor import GNNPredictor
        _predictor = GNNPredictor()
        return _predictor
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# EMPIRICALLY TUNED SCALING
# Based on actual benchmark results:
#   gen=130-150 is sweet spot, higher pop adds runtime not quality
#   0.97 cooling slightly better than 0.95 for hybrid SA phase
# ─────────────────────────────────────────────────────────────

def scale_iterations(gate_count: int) -> int:
    return min(200, max(10, gate_count // 50))


def scale_ga_params(gate_count: int) -> dict:
    """
    Empirically tuned tiers:
      <300    : pop=20, gen=80
      300-600 : pop=25, gen=140  ← sweet spot for s1196
      600-1500: pop=25, gen=150
      1500-5k : pop=30, gen=160
      >5000   : pop=30, gen=150  (runtime cap)
    """
    if gate_count < 300:
        p, g = 20, 80
    elif gate_count < 600:
        p, g = 25, 140
    elif gate_count < 1500:
        p, g = 25, 150
    elif gate_count < 5000:
        p, g = 30, 160
    else:
        p, g = 30, 150
    return dict(population_size=p, generations=g,
                survival_rate=0.3, mutation_rate=0.4,
                validate=True, verbose=False)


def build_configs(gate_count: int) -> dict:
    iters  = scale_iterations(gate_count)
    ga_cfg = scale_ga_params(gate_count)
    return {
        "sa": dict(
            initial_temp=100.0, cooling_rate=0.95,
            min_temp=0.1, iterations_per_temp=iters,
            validate=True, verbose=False,
        ),
        "ga": ga_cfg,
        "hybrid": dict(
            ga_population=ga_cfg['population_size'],
            ga_generations=ga_cfg['generations'],
            ga_survival=0.3, ga_mutation=0.4,
            sa_initial_temp=100.0, sa_cooling=0.97,
            sa_min_temp=0.1, sa_iterations=iters,
            verbose=False,
        ),
        "gnn_sa": dict(
            initial_temp=100.0, cooling_rate=0.95,
            min_temp=0.1, iterations_per_temp=iters,
            verify_every=20, validate=True, verbose=False,
        ),
    }


# GNN reliable range — update after retraining
GNN_TRAINED_MIN = 200
GNN_TRAINED_MAX = 1500


def select_optimizer(gate_count: int, force: str = None) -> str:
    if force and force in ("sa", "ga", "hybrid", "gnn_sa"):
        return force
    predictor = _get_predictor()
    if gate_count < 500:
        return "hybrid"
    elif (predictor is not None
          and GNN_TRAINED_MIN <= gate_count <= GNN_TRAINED_MAX):
        return "gnn_sa"
    else:
        return "hybrid"


def _save_output(circuit, best_gates, best_cost,
                  optimizer_name, elapsed):
    """Writes .bench and report to output/ folder."""
    try:
        from optimizer.bench_writer import save_optimized_circuit
        os.makedirs("output", exist_ok=True)
        bench_path, report_path = save_optimized_circuit(
            output_dir       = "output",
            original_circuit = circuit,
            optimized_gates  = best_gates,
            optimized_cost   = best_cost,
            optimizer_name   = optimizer_name,
            time_seconds     = elapsed,
        )
        print(f"  Saved  : {bench_path}")
        print(f"  Report : {report_path}")
        return bench_path, report_path
    except Exception as e:
        print(f"  [bench_writer] {e}")
        return None, None


# ─────────────────────────────────────────────────────────────
# RUN ALL OPTIMIZERS — PICK BEST
# ─────────────────────────────────────────────────────────────

def optimize_all(circuit, verbose: bool = True) -> dict:
    """
    Runs SA, GA, Hybrid, GNN-SA (if available + in trained range).
    Picks best result, writes output .bench + report.

    Note on large circuits (>5000 gates):
      SA/GA/Hybrid work but are slow (~20-30 min on CPU).
      GNN-SA is skipped — not reliable outside training range.
      For 20k+ circuits, SA alone is the most practical.
    """
    from optimizer.simulated_annealing import simulated_annealing
    from optimizer.genetic_algorithm   import genetic_algorithm
    from optimizer.hybrid_optimizer    import hybrid_optimize

    gate_count = circuit.gate_count
    configs    = build_configs(gate_count)
    ga_params  = scale_ga_params(gate_count)
    iters      = scale_iterations(gate_count)
    predictor  = _get_predictor()

    if verbose:
        print("=" * 62)
        print("  FULL OPTIMIZER COMPARISON")
        print("=" * 62)
        print(f"  Circuit  : {circuit.name}  ({gate_count} gates)")
        print(f"  Original : {round(circuit.cost, 4)}")
        print(f"  SA iters : {iters}/step")
        print(f"  GA       : pop={ga_params['population_size']}  "
              f"gen={ga_params['generations']}")
        if gate_count > 5000:
            print(f"  NOTE     : Large circuit — GNN-SA skipped, "
                  f"expect slow runtime")
        print("-" * 62)

    results = {}

    def _run(label, fn, *args, **kwargs):
        print(f"  [{label:<7}] ...", end=" ", flush=True)
        t0         = time.perf_counter()
        g, cost, _ = fn(*args, **kwargs)
        elapsed    = round(time.perf_counter() - t0, 2)
        imp        = round((circuit.cost - cost) / circuit.cost * 100, 2)
        results[label] = dict(gates=g, cost=cost, imp=imp, time=elapsed)
        print(f"{imp:6.2f}%  {elapsed}s")

    _run("SA",     simulated_annealing, circuit, **configs["sa"])
    _run("GA",     genetic_algorithm,   circuit, **configs["ga"])
    _run("Hybrid", hybrid_optimize,     circuit, **configs["hybrid"])

    if (predictor is not None
            and GNN_TRAINED_MIN <= gate_count <= GNN_TRAINED_MAX):
        from optimizer.gnn_optimizer import gnn_simulated_annealing
        _run("GNN-SA", gnn_simulated_annealing,
             circuit, predictor, **configs["gnn_sa"])
    else:
        results["GNN-SA"] = None
        reason = ("no model" if predictor is None
                  else "outside trained range")
        if verbose:
            print(f"  [GNN-SA ] skipped ({reason})")

    best_name = min(
        (k for k in results if results[k] is not None),
        key=lambda k: results[k]['cost']
    )
    best = results[best_name]

    if verbose:
        print("-" * 62)
        print(f"  WINNER : {best_name}  "
              f"improvement={best['imp']}%  "
              f"time={best['time']}s")
        print("=" * 62)

    _save_output(circuit, best['gates'], best['cost'],
                  best_name, best['time'])

    return {
        "circuit_name"  : circuit.name,
        "gate_count"    : gate_count,
        "original_cost" : circuit.cost,
        "results"       : {
            k: {"cost": v['cost'], "improvement": v['imp'],
                "time": v['time']}
            if v else None
            for k, v in results.items()
        },
        "best_optimizer": best_name,
        "best_cost"     : best['cost'],
        "best_imp"      : best['imp'],
    }


# ─────────────────────────────────────────────────────────────
# SINGLE OPTIMIZER (used by FastAPI)
# ─────────────────────────────────────────────────────────────

def optimize(circuit, force_optimizer: str = None,
             verbose: bool = True) -> dict:
    from optimizer.simulated_annealing import simulated_annealing
    from optimizer.genetic_algorithm   import genetic_algorithm
    from optimizer.hybrid_optimizer    import hybrid_optimize

    gate_count = circuit.gate_count
    optimizer  = select_optimizer(gate_count, force_optimizer)
    configs    = build_configs(gate_count)
    iters      = scale_iterations(gate_count)

    if verbose:
        print(f"  [{circuit.name}] {gate_count} gates → {optimizer.upper()}")

    t0 = time.perf_counter()

    if optimizer == "sa":
        best_gates, cost, report = simulated_annealing(
            circuit, **configs["sa"])
    elif optimizer == "ga":
        best_gates, cost, report = genetic_algorithm(
            circuit, **configs["ga"])
    elif optimizer == "hybrid":
        best_gates, cost, report = hybrid_optimize(
            circuit, **configs["hybrid"])
    elif optimizer == "gnn_sa":
        predictor = _get_predictor()
        if predictor is None:
            optimizer  = "hybrid"
            best_gates, cost, report = hybrid_optimize(
                circuit, **configs["hybrid"])
        else:
            from optimizer.gnn_optimizer import gnn_simulated_annealing
            best_gates, cost, report = gnn_simulated_annealing(
                circuit, predictor, **configs["gnn_sa"])

    elapsed     = round(time.perf_counter() - t0, 3)
    improvement = round((circuit.cost - cost) / circuit.cost * 100, 4)

    if verbose:
        print(f"  Improvement : {improvement}%  Time: {elapsed}s")

    # Always write output files
    _save_output(circuit, best_gates, cost, optimizer, elapsed)

    return {
        "circuit_name"   : circuit.name,
        "gate_count"     : gate_count,
        "original_cost"  : round(circuit.cost, 4),
        "optimized_cost" : round(cost, 4),
        "improvement_pct": improvement,
        "optimizer_used" : optimizer,
        "iterations_used": iters,
        "time_seconds"   : elapsed,
        "detail"         : report,
    }


def optimize_file(filepath: str, force_optimizer: str = None,
                   verbose: bool = True) -> dict:
    from core.pipeline import load_circuit
    circuit, _ = load_circuit(filepath)
    return optimize(circuit, force_optimizer, verbose)


# ─────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Parameter scaling:")
    print(f"  {'Gates':>8}  {'Pop':>5}  {'Gen':>5}  {'SA_iters':>10}")
    for n in [6, 292, 547, 659, 1193, 3512, 20679]:
        p = scale_ga_params(n)
        print(f"  {n:>8}  {p['population_size']:>5}  "
              f"{p['generations']:>5}  {scale_iterations(n):>10}")

    print()
    # Quick single-optimizer test
    optimize_file("data/benchmarks/s1196.bench",
                   force_optimizer="hybrid", verbose=True)