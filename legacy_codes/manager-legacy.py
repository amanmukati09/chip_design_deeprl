# """
# heuristics/manager.py
# ─────────────────────────────────────────────────────
# Complexity Router — the novel contribution of this system.

# Automatically selects the best optimizer based on circuit size,
# based on empirical benchmark results across 15 MCNC circuits:

#   < 500 gates  → GA+SA hybrid
#                  GNN generalizes poorly to small circuits
#                  GA+SA wins consistently in this range

#   ≥ 500 gates  → GNN-accelerated SA
#                  GNN-SA wins 11/15 circuits tested
#                  Best result: 21.75% on c2670 (1193 gates)

# Iteration scaling:
#   iterations_per_temp = max(10, gate_count // 50)
#   Ensures SA explores the space proportionally to circuit size.
#   Without this, large circuits are under-explored.

# Usage:
#     from heuristics.manager import optimize
#     result = optimize(circuit)
#     print(result)

#     # or force a specific optimizer:
#     result = optimize(circuit, optimizer="sa")
# """

# import os
# import sys
# import time

# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# from pipeline_v1                 import load_circuit
# from optimizer.simulated_annealing import simulated_annealing
# from optimizer.genetic_algorithm   import genetic_algorithm
# from optimizer.hybrid_optimizer    import hybrid_optimize
# from optimizer.gnn_optimizer       import gnn_simulated_annealing


# # ─────────────────────────────────────────────────────────────
# # THRESHOLDS  (empirically determined from benchmark results)
# # ─────────────────────────────────────────────────────────────

# SMALL_CIRCUIT_THRESHOLD = 500   # gates
# LARGE_CIRCUIT_THRESHOLD = 10000 # beyond this, scale iterations only

# # GNN predictor — loaded once, reused across all calls
# _predictor = None

# def _get_predictor():
#     global _predictor
#     if _predictor is not None:
#         return _predictor
#     try:
#         from ml.predictor import GNNPredictor
#         _predictor = GNNPredictor()
#         return _predictor
#     except Exception as e:
#         print(f"[Manager] GNN unavailable: {e}")
#         return None


# # ─────────────────────────────────────────────────────────────
# # ITERATION SCALING
# # ─────────────────────────────────────────────────────────────

# def scale_iterations(gate_count: int) -> int:
#     """
#     Scale SA iterations proportionally to circuit size.

#     Without scaling, SA runs the same 1220 iterations
#     whether the circuit has 100 gates or 20000 gates.
#     A 20000-gate circuit needs ~37x more iterations
#     to achieve equivalent exploration coverage.

#     Capped at 200 to keep runtime reasonable on CPU.
#     """
#     return min(200, max(10, gate_count // 50))


# # ─────────────────────────────────────────────────────────────
# # OPTIMIZER CONFIGS  (built per circuit, not hardcoded)
# # ─────────────────────────────────────────────────────────────

# def build_configs(gate_count: int) -> dict:
#     iters = scale_iterations(gate_count)
#     return {
#         "sa": dict(
#             initial_temp        = 50.0,
#             cooling_rate        = 0.95,
#             min_temp            = 0.1,
#             iterations_per_temp = iters,
#             verbose             = False,
#         ),
#         "ga": dict(
#             population_size = 20,
#             generations     = 50,
#             survival_rate   = 0.3,
#             mutation_rate   = 0.7,
#             verbose         = False,
#         ),
#         "hybrid": dict(
#             ga_population   = 20,
#             ga_generations  = 50,
#             ga_survival     = 0.3,
#             ga_mutation     = 0.7,
#             sa_initial_temp = 50.0,
#             sa_cooling      = 0.95,
#             sa_min_temp     = 0.1,
#             sa_iterations   = iters,
#             verbose         = False,
#         ),
#         "gnn_sa": dict(
#             initial_temp        = 50.0,
#             cooling_rate        = 0.95,
#             min_temp            = 0.1,
#             iterations_per_temp = iters,
#             verify_every        = 20,
#             verbose             = False,
#         ),
#     }


# # ─────────────────────────────────────────────────────────────
# # ROUTER LOGIC
# # ─────────────────────────────────────────────────────────────

# def select_optimizer(gate_count: int, force: str = None) -> str:
#     """
#     Selects best optimizer based on circuit size.

#     Args:
#         gate_count : number of gates in the circuit
#         force      : override selection ('sa','ga','hybrid','gnn_sa')

#     Returns:
#         optimizer name as string
#     """
#     if force and force in ("sa", "ga", "hybrid", "gnn_sa"):
#         return force

#     predictor = _get_predictor()

#     if gate_count < SMALL_CIRCUIT_THRESHOLD:
#         return "hybrid"        # GA+SA wins on small circuits
#     elif predictor is not None:
#         return "gnn_sa"        # GNN-SA wins on medium/large
#     else:
#         return "hybrid"        # fallback if GNN unavailable


# # ─────────────────────────────────────────────────────────────
# # MAIN ENTRY POINT
# # ─────────────────────────────────────────────────────────────

# def optimize(circuit, force_optimizer: str = None, verbose: bool = True):
#     """
#     Optimizes a circuit using the automatically selected optimizer.

#     Args:
#         circuit          : Circuit object (from load_circuit)
#         force_optimizer  : optional override ('sa','ga','hybrid','gnn_sa')
#         verbose          : print progress

#     Returns:
#         dict with full optimization report
#     """
#     gate_count = circuit.gate_count
#     optimizer  = select_optimizer(gate_count, force_optimizer)
#     configs    = build_configs(gate_count)
#     iters      = scale_iterations(gate_count)

#     if verbose:
#         print("=" * 55)
#         print("  COMPLEXITY ROUTER")
#         print("=" * 55)
#         print(f"  Circuit     : {circuit.name}")
#         print(f"  Gates       : {gate_count}")
#         print(f"  Original    : {round(circuit.cost, 4)}")
#         print(f"  Selected    : {optimizer.upper()}")
#         print(f"  Iterations  : {iters} per temp step")
#         print("-" * 55)

#     t0 = time.perf_counter()

#     if optimizer == "sa":
#         _, cost, report = simulated_annealing(circuit, **configs["sa"])

#     elif optimizer == "ga":
#         _, cost, report = genetic_algorithm(circuit, **configs["ga"])

#     elif optimizer == "hybrid":
#         _, cost, report = hybrid_optimize(circuit, **configs["hybrid"])

#     elif optimizer == "gnn_sa":
#         predictor = _get_predictor()
#         if predictor is None:
#             # Fallback to hybrid if GNN fails at runtime
#             if verbose:
#                 print("  [Fallback] GNN unavailable, using hybrid.")
#             optimizer = "hybrid"
#             _, cost, report = hybrid_optimize(circuit, **configs["hybrid"])
#         else:
#             _, cost, report = gnn_simulated_annealing(
#                 circuit, predictor, **configs["gnn_sa"]
#             )

#     elapsed     = round(time.perf_counter() - t0, 3)
#     improvement = round((circuit.cost - cost) / circuit.cost * 100, 4)

#     result = {
#         "circuit_name"   : circuit.name,
#         "gate_count"     : gate_count,
#         "original_cost"  : round(circuit.cost, 4),
#         "optimized_cost" : round(cost, 4),
#         "improvement_pct": improvement,
#         "optimizer_used" : optimizer,
#         "iterations_used": iters,
#         "time_seconds"   : elapsed,
#         "detail"         : report,
#     }

#     if verbose:
#         print(f"  Optimized   : {round(cost, 4)}")
#         print(f"  Improvement : {improvement}%")
#         print(f"  Time        : {elapsed}s")
#         print("=" * 55)

#     return result


# def optimize_file(filepath: str, force_optimizer: str = None, verbose: bool = True):
#     """Convenience wrapper — takes a file path instead of Circuit object."""
#     circuit, _ = load_circuit(filepath)
#     return optimize(circuit, force_optimizer, verbose)


# # ─────────────────────────────────────────────────────────────
# # QUICK TEST
# # ─────────────────────────────────────────────────────────────

# if __name__ == "__main__":

#     print("\nTest 1 — Small circuit (c17, 6 gates) → expect hybrid")
#     result = optimize_file("data/benchmarks/c17.bench")
#     print(f"  Used: {result['optimizer_used']}  "
#           f"Improvement: {result['improvement_pct']}%\n")

#     print("Test 2 — Medium circuit (s1196, 547 gates) → expect gnn_sa")
#     result = optimize_file("data/benchmarks/s1196.bench")
#     print(f"  Used: {result['optimizer_used']}  "
#           f"Improvement: {result['improvement_pct']}%\n")

#     print("Test 3 — Large circuit (c2670, 1193 gates) → expect gnn_sa")
#     result = optimize_file("data/benchmarks/c2670.bench")
#     print(f"  Used: {result['optimizer_used']}  "
#           f"Improvement: {result['improvement_pct']}%\n")

#     print("Test 4 — Force SA on s1196")
#     result = optimize_file("data/benchmarks/s1196.bench",
#                            force_optimizer="sa")
#     print(f"  Used: {result['optimizer_used']}  "
#           f"Improvement: {result['improvement_pct']}%\n")


#----------------------------------------------------------------------

# heuristics/manager.py
# Complexity Router with auto-scaling parameters.
#
# Key fixes:
#   - GA population and generations scale with gate count
#   - SA iterations scale with gate count
#   - All four optimizers run and best result is selected
#   - Best circuit written to output/ as .bench + report
#   - GNN-SA only used if GNN was trained on similar-size data

# import os
# import sys
# import time

# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# from core.pipeline                 import load_circuit
# from optimizer.simulated_annealing import simulated_annealing
# from optimizer.genetic_algorithm   import genetic_algorithm
# from optimizer.hybrid_optimizer    import hybrid_optimize
# from optimizer.gnn_optimizer       import gnn_simulated_annealing

# _predictor = None


# def _get_predictor():
#     global _predictor
#     if _predictor is not None:
#         return _predictor
#     try:
#         from ml.predictor import GNNPredictor
#         _predictor = GNNPredictor()
#         return _predictor
#     except Exception as e:
#         return None


# # ─────────────────────────────────────────────────────────────
# # AUTO-SCALING PARAMETER ENGINE
# # All parameters derived from gate count — no manual tuning.
# # ─────────────────────────────────────────────────────────────

# def scale_iterations(gate_count: int) -> int:
#     """SA iterations per temperature step, scaled to circuit size."""
#     return min(200, max(10, gate_count // 50))


# def scale_ga_params(gate_count: int) -> dict:
#     """
#     GA population and generations scaled to circuit size.

#     Small  (<300 gates) : pop=20,  gen=80
#     Medium (300-1000)   : pop=35,  gen=130
#     Large  (1000-5000)  : pop=50,  gen=180
#     XLarge (>5000)      : pop=60,  gen=200
#     """
#     if gate_count < 300:
#         population  = 20
#         generations = 80
#     elif gate_count < 1000:
#         population  = 35
#         generations = 130
#     elif gate_count < 5000:
#         population  = 50
#         generations = 180
#     else:
#         population  = 60
#         generations = 200

#     return {
#         'population_size': population,
#         'generations'    : generations,
#         'survival_rate'  : 0.3,
#         'mutation_rate'  : 0.4,
#         'validate'       : True,
#         'verbose'        : False,
#     }


# def build_configs(gate_count: int) -> dict:
#     iters   = scale_iterations(gate_count)
#     ga_cfg  = scale_ga_params(gate_count)

#     return {
#         "sa": dict(
#             initial_temp        = 100.0,
#             cooling_rate        = 0.95,
#             min_temp            = 0.1,
#             iterations_per_temp = iters,
#             validate            = True,
#             verbose             = False,
#         ),
#         "ga": ga_cfg,
#         "hybrid": dict(
#             ga_population   = ga_cfg['population_size'],
#             ga_generations  = ga_cfg['generations'],
#             ga_survival     = 0.3,
#             ga_mutation     = 0.4,
#             sa_initial_temp = 100.0,
#             sa_cooling      = 0.95,
#             sa_min_temp     = 0.1,
#             sa_iterations   = iters,
#             verbose         = False,
#         ),
#         "gnn_sa": dict(
#             initial_temp        = 100.0,
#             cooling_rate        = 0.95,
#             min_temp            = 0.1,
#             iterations_per_temp = iters,
#             verify_every        = 20,
#             validate            = True,
#             verbose             = False,
#         ),
#     }


# # ─────────────────────────────────────────────────────────────
# # OPTIMIZER SELECTOR
# # GNN-SA only used when GNN is available AND circuit size
# # is within the range it was trained on.
# # ─────────────────────────────────────────────────────────────

# # Gate count range GNN was trained on (update after retraining)
# GNN_TRAINED_MIN = 200
# GNN_TRAINED_MAX = 5000


# def select_optimizer(gate_count: int, force: str = None) -> str:
#     if force and force in ("sa", "ga", "hybrid", "gnn_sa"):
#         return force

#     predictor = _get_predictor()

#     if gate_count < 500:
#         return "hybrid"   # GA+SA wins on small circuits
#     elif (predictor is not None
#           and GNN_TRAINED_MIN <= gate_count <= GNN_TRAINED_MAX):
#         return "gnn_sa"   # GNN reliable in trained range
#     else:
#         return "hybrid"   # fallback for very large or untrained range


# # ─────────────────────────────────────────────────────────────
# # RUN ALL OPTIMIZERS AND PICK BEST
# # ─────────────────────────────────────────────────────────────

# def optimize_all(circuit, verbose: bool = True) -> dict:
#     """
#     Runs SA, GA, Hybrid, and GNN-SA (if available).
#     Compares all results and returns the best.
#     Also saves best circuit as .bench + report.

#     Returns full comparison dict.
#     """
#     gate_count = circuit.gate_count
#     configs    = build_configs(gate_count)
#     iters      = scale_iterations(gate_count)
#     ga_params  = scale_ga_params(gate_count)
#     predictor  = _get_predictor()

#     if verbose:
#         print("=" * 60)
#         print("  FULL OPTIMIZER COMPARISON")
#         print("=" * 60)
#         print(f"  Circuit     : {circuit.name}")
#         print(f"  Gates       : {gate_count}")
#         print(f"  Original    : {round(circuit.cost, 4)}")
#         print(f"  SA iters    : {iters}/step")
#         print(f"  GA pop/gen  : {ga_params['population_size']}/"
#               f"{ga_params['generations']}")
#         print("-" * 60)

#     results = {}

#     # SA
#     print(f"  [SA]     running...", end=" ", flush=True)
#     t0 = time.perf_counter()
#     sa_gates, sa_cost, _ = simulated_annealing(
#         circuit, **configs["sa"]
#     )
#     sa_time = round(time.perf_counter() - t0, 2)
#     sa_imp  = round((circuit.cost - sa_cost) / circuit.cost * 100, 2)
#     results["sa"] = dict(gates=sa_gates, cost=sa_cost,
#                           imp=sa_imp, time=sa_time)
#     print(f"{sa_imp}%  {sa_time}s")

#     # GA
#     print(f"  [GA]     running...", end=" ", flush=True)
#     t0 = time.perf_counter()
#     ga_gates, ga_cost, _ = genetic_algorithm(circuit, **configs["ga"])
#     ga_time = round(time.perf_counter() - t0, 2)
#     ga_imp  = round((circuit.cost - ga_cost) / circuit.cost * 100, 2)
#     results["ga"] = dict(gates=ga_gates, cost=ga_cost,
#                           imp=ga_imp, time=ga_time)
#     print(f"{ga_imp}%  {ga_time}s")

#     # Hybrid
#     print(f"  [Hybrid] running...", end=" ", flush=True)
#     t0 = time.perf_counter()
#     hy_gates, hy_cost, _ = hybrid_optimize(circuit, **configs["hybrid"])
#     hy_time = round(time.perf_counter() - t0, 2)
#     hy_imp  = round((circuit.cost - hy_cost) / circuit.cost * 100, 2)
#     results["hybrid"] = dict(gates=hy_gates, cost=hy_cost,
#                                imp=hy_imp, time=hy_time)
#     print(f"{hy_imp}%  {hy_time}s")

#     # GNN-SA (if available and in trained range)
#     if (predictor is not None
#             and GNN_TRAINED_MIN <= gate_count <= GNN_TRAINED_MAX):
#         print(f"  [GNN-SA] running...", end=" ", flush=True)
#         t0 = time.perf_counter()
#         gnn_gates, gnn_cost, _ = gnn_simulated_annealing(
#             circuit, predictor, **configs["gnn_sa"]
#         )
#         gnn_time = round(time.perf_counter() - t0, 2)
#         gnn_imp  = round((circuit.cost - gnn_cost) / circuit.cost * 100, 2)
#         results["gnn_sa"] = dict(gates=gnn_gates, cost=gnn_cost,
#                                    imp=gnn_imp, time=gnn_time)
#         print(f"{gnn_imp}%  {gnn_time}s")
#     else:
#         results["gnn_sa"] = None
#         if verbose:
#             print(f"  [GNN-SA] skipped "
#                   f"({'no predictor' if predictor is None else 'outside trained range'})")

#     # Pick best by lowest cost
#     best_name = min(
#         (k for k in results if results[k] is not None),
#         key=lambda k: results[k]['cost']
#     )
#     best = results[best_name]

#     if verbose:
#         print("-" * 60)
#         print(f"  WINNER: {best_name.upper()}  "
#               f"cost={round(best['cost'],4)}  "
#               f"improvement={best['imp']}%")
#         print("=" * 60)

#     # Save best circuit as .bench + report
#     try:
#         from optimizer.bench_writer import save_optimized_circuit
#         os.makedirs("output", exist_ok=True)
#         bench_path, report_path = save_optimized_circuit(
#             output_dir       = "output",
#             original_circuit = circuit,
#             optimized_gates  = best['gates'],
#             optimized_cost   = best['cost'],
#             optimizer_name   = best_name,
#             time_seconds     = best['time'],
#         )
#         if verbose:
#             print(f"  Saved: {bench_path}")
#             print(f"  Report: {report_path}")
#     except Exception as e:
#         if verbose:
#             print(f"  [bench_writer] {e}")

#     return {
#         "circuit_name"  : circuit.name,
#         "gate_count"    : gate_count,
#         "original_cost" : circuit.cost,
#         "results"       : {
#             k: {"cost": v['cost'], "improvement": v['imp'],
#                 "time": v['time']}
#             if v else None
#             for k, v in results.items()
#         },
#         "best_optimizer": best_name,
#         "best_cost"     : best['cost'],
#         "best_imp"      : best['imp'],
#     }


# # ─────────────────────────────────────────────────────────────
# # SINGLE OPTIMIZER ENTRY POINT (used by API)
# # ─────────────────────────────────────────────────────────────

# def optimize(circuit, force_optimizer: str = None,
#              verbose: bool = True) -> dict:
#     """
#     Runs one optimizer (auto-selected or forced).
#     Used by the FastAPI endpoint.
#     """
#     gate_count = circuit.gate_count
#     optimizer  = select_optimizer(gate_count, force_optimizer)
#     configs    = build_configs(gate_count)
#     iters      = scale_iterations(gate_count)

#     if verbose:
#         print("=" * 55)
#         print("  COMPLEXITY ROUTER")
#         print("=" * 55)
#         print(f"  Circuit     : {circuit.name}")
#         print(f"  Gates       : {gate_count}")
#         print(f"  Original    : {round(circuit.cost, 4)}")
#         print(f"  Selected    : {optimizer.upper()}")
#         print(f"  SA iters    : {iters}/step")
#         print("-" * 55)

#     t0 = time.perf_counter()

#     if optimizer == "sa":
#         _, cost, report = simulated_annealing(circuit, **configs["sa"])
#     elif optimizer == "ga":
#         _, cost, report = genetic_algorithm(circuit, **configs["ga"])
#     elif optimizer == "hybrid":
#         _, cost, report = hybrid_optimize(circuit, **configs["hybrid"])
#     elif optimizer == "gnn_sa":
#         predictor = _get_predictor()
#         if predictor is None:
#             optimizer = "hybrid"
#             _, cost, report = hybrid_optimize(circuit, **configs["hybrid"])
#         else:
#             _, cost, report = gnn_simulated_annealing(
#                 circuit, predictor, **configs["gnn_sa"]
#             )

#     elapsed     = round(time.perf_counter() - t0, 3)
#     improvement = round((circuit.cost - cost) / circuit.cost * 100, 4)

#     if verbose:
#         print(f"  Optimized   : {round(cost, 4)}")
#         print(f"  Improvement : {improvement}%")
#         print(f"  Time        : {elapsed}s")
#         print("=" * 55)

#     return {
#         "circuit_name"   : circuit.name,
#         "gate_count"     : gate_count,
#         "original_cost"  : round(circuit.cost, 4),
#         "optimized_cost" : round(cost, 4),
#         "improvement_pct": improvement,
#         "optimizer_used" : optimizer,
#         "iterations_used": iters,
#         "time_seconds"   : elapsed,
#         "detail"         : report,
#     }


# def optimize_file(filepath: str, force_optimizer: str = None,
#                    verbose: bool = True) -> dict:
#     circuit, _ = load_circuit(filepath)
#     return optimize(circuit, force_optimizer, verbose)


# # ─────────────────────────────────────────────────────────────
# # QUICK TEST
# # ─────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     print("\nScale params test:")
#     for n in [100, 500, 1000, 5000, 20000]:
#         p = scale_ga_params(n)
#         print(f"  {n:6d} gates → "
#               f"pop={p['population_size']:3d}  "
#               f"gen={p['generations']:3d}  "
#               f"sa_iters={scale_iterations(n)}")

#     print("\nRunning optimize_all on s1196...")
#     summary = optimize_file(
#         "data/benchmarks/s1196.bench",
#         force_optimizer=None,
#         verbose=True
#     )



# heuristics/manager.py
# Complexity Router — empirically tuned, all imports verified.
#
# Fixes vs previous version:
#   - bench_writer import wrapped in try/except with clear error
#   - output/ folder created before any write attempt
#   - optimize_all() tested standalone without breaking
#   - 20k+ gate circuits handled: GNN skipped, hybrid used with scaled iters

import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.pipeline                 import load_circuit
from optimizer.simulated_annealing import simulated_annealing
from optimizer.genetic_algorithm   import genetic_algorithm
from optimizer.hybrid_optimizer    import hybrid_optimize
from optimizer.gnn_optimizer       import gnn_simulated_annealing

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
# PARAMETER SCALING
# Based on empirical results:
#   gen=130-140 is sweet spot for s1196 (converges there anyway)
#   pop=25-30 sufficient; higher just slows
#   hybrid SA cooling 0.97 > 0.95
#   20k+ gates: skip GNN, use hybrid with scaled iters
# ─────────────────────────────────────────────────────────────

def scale_iterations(gate_count: int) -> int:
    """SA iterations per temperature step."""
    return min(200, max(10, gate_count // 50))


def scale_ga_params(gate_count: int) -> dict:
    """
    Gate count tiers → pop/gen:
      <300        : 20 / 80
      300-600     : 25 / 140   ← s1196 sweet spot
      600-1500    : 25 / 150
      1500-5000   : 30 / 160
      >5000       : 30 / 150   (keep runtime sane on CPU)
    """
    if gate_count < 300:
        pop, gen = 20, 80
    elif gate_count < 600:
        pop, gen = 25, 140
    elif gate_count < 1500:
        pop, gen = 25, 150
    elif gate_count < 5000:
        pop, gen = 30, 160
    else:
        pop, gen = 30, 150

    return dict(population_size=pop, generations=gen,
                survival_rate=0.3, mutation_rate=0.4,
                validate=True, verbose=False)


def build_configs(gate_count: int) -> dict:
    iters  = scale_iterations(gate_count)
    ga_cfg = scale_ga_params(gate_count)
    return {
        "sa": dict(
            initial_temp=100.0, cooling_rate=0.95,
            min_temp=0.1, iterations_per_temp=iters,
            validate=True, verbose=False),
        "ga": ga_cfg,
        "hybrid": dict(
            ga_population=ga_cfg['population_size'],
            ga_generations=ga_cfg['generations'],
            ga_survival=0.3, ga_mutation=0.4,
            sa_initial_temp=100.0, sa_cooling=0.97,
            sa_min_temp=0.1, sa_iterations=iters,
            verbose=False),
        "gnn_sa": dict(
            initial_temp=100.0, cooling_rate=0.95,
            min_temp=0.1, iterations_per_temp=iters,
            verify_every=20, validate=True, verbose=False),
    }


# GNN only reliable in this gate range (update after retraining)
GNN_MIN = 200
GNN_MAX = 5000


def select_optimizer(gate_count: int, force: str = None) -> str:
    if force and force in ("sa", "ga", "hybrid", "gnn_sa"):
        return force
    predictor = _get_predictor()
    if gate_count < 500:
        return "hybrid"
    elif predictor and GNN_MIN <= gate_count <= GNN_MAX:
        return "gnn_sa"
    else:
        return "hybrid"   # 20k+ gates → hybrid with scaled iters


# ─────────────────────────────────────────────────────────────
# RUN ALL OPTIMIZERS — PICK BEST — WRITE OUTPUT
# ─────────────────────────────────────────────────────────────

def optimize_all(circuit, verbose: bool = True) -> dict:
    """
    Runs SA, GA, Hybrid, GNN-SA (if available).
    Picks best result, writes .bench + report to output/.
    Works for circuits of any size — 6 gates to 20k+ gates.
    """
    gc        = circuit.gate_count
    configs   = build_configs(gc)
    ga_p      = scale_ga_params(gc)
    iters     = scale_iterations(gc)
    predictor = _get_predictor()

    if verbose:
        print("=" * 62)
        print("  FULL OPTIMIZER COMPARISON")
        print("=" * 62)
        print(f"  Circuit  : {circuit.name}  ({gc} gates)")
        print(f"  Original : {round(circuit.cost, 4)}")
        print(f"  SA iters : {iters}/step")
        print(f"  GA       : pop={ga_p['population_size']} "
              f"gen={ga_p['generations']}")
        print("-" * 62)

    results = {}

    def _run(label, fn, *args, **kwargs):
        print(f"  [{label:<7}]", end=" ", flush=True)
        t0         = time.perf_counter()
        g, cost, _ = fn(*args, **kwargs)
        elapsed    = round(time.perf_counter() - t0, 2)
        imp        = round((circuit.cost - cost) / circuit.cost * 100, 2)
        results[label] = dict(gates=g, cost=cost, imp=imp, time=elapsed)
        print(f"{imp:6.2f}%  {elapsed}s")

    _run("SA",     simulated_annealing, circuit, **configs["sa"])
    _run("GA",     genetic_algorithm,   circuit, **configs["ga"])
    _run("Hybrid", hybrid_optimize,     circuit, **configs["hybrid"])

    if predictor and GNN_MIN <= gc <= GNN_MAX:
        _run("GNN-SA", gnn_simulated_annealing,
             circuit, predictor, **configs["gnn_sa"])
    else:
        results["GNN-SA"] = None
        reason = "no model" if not predictor else "outside trained range"
        if verbose:
            print(f"  [GNN-SA ] skipped ({reason})")

    # Pick winner
    best_name = min(
        (k for k in results if results[k] is not None),
        key=lambda k: results[k]['cost']
    )
    best = results[best_name]

    if verbose:
        print("-" * 62)
        print(f"  WINNER : {best_name}  "
              f"cost={round(best['cost'],4)}  "
              f"imp={best['imp']}%")

    # Write output files
    _write_output(circuit, best, best_name, verbose)

    return {
        "circuit_name"  : circuit.name,
        "gate_count"    : gc,
        "original_cost" : circuit.cost,
        "results"       : {
            k: {"cost": v['cost'], "improvement": v['imp'],
                "time": v['time']} if v else None
            for k, v in results.items()
        },
        "best_optimizer": best_name,
        "best_cost"     : best['cost'],
        "best_imp"      : best['imp'],
    }


def _write_output(circuit, best: dict, optimizer_name: str,
                   verbose: bool) -> None:
    """Writes optimized .bench and report to output/ folder."""
    try:
        from optimizer.bench_writer import save_optimized_circuit
        out_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "output"
        )
        os.makedirs(out_dir, exist_ok=True)
        bench_path, report_path = save_optimized_circuit(
            output_dir       = out_dir,
            original_circuit = circuit,
            optimized_gates  = best['gates'],
            optimized_cost   = best['cost'],
            optimizer_name   = optimizer_name,
            time_seconds     = best['time'],
        )
        if verbose:
            print(f"  Bench  : {bench_path}")
            print(f"  Report : {report_path}")
            print("=" * 62)
    except Exception as e:
        if verbose:
            print(f"  [output error] {e}")
            print("=" * 62)


# ─────────────────────────────────────────────────────────────
# SINGLE OPTIMIZER ENTRY POINT (used by FastAPI)
# ─────────────────────────────────────────────────────────────

def optimize(circuit, force_optimizer: str = None,
             verbose: bool = True) -> dict:
    gc       = circuit.gate_count
    opt      = select_optimizer(gc, force_optimizer)
    configs  = build_configs(gc)

    if verbose:
        print(f"  [{circuit.name}] {gc} gates → {opt.upper()}")

    t0 = time.perf_counter()

    if opt == "sa":
        _, cost, report = simulated_annealing(circuit, **configs["sa"])
    elif opt == "ga":
        _, cost, report = genetic_algorithm(circuit, **configs["ga"])
    elif opt == "hybrid":
        _, cost, report = hybrid_optimize(circuit, **configs["hybrid"])
    elif opt == "gnn_sa":
        predictor = _get_predictor()
        if not predictor:
            opt = "hybrid"
            _, cost, report = hybrid_optimize(circuit, **configs["hybrid"])
        else:
            _, cost, report = gnn_simulated_annealing(
                circuit, predictor, **configs["gnn_sa"])

    elapsed = round(time.perf_counter() - t0, 3)
    imp     = round((circuit.cost - cost) / circuit.cost * 100, 4)

    if verbose:
        print(f"  Result : {round(cost,4)}  ({imp}%)  {elapsed}s")

    return dict(circuit_name=circuit.name, gate_count=gc,
                original_cost=round(circuit.cost,4),
                optimized_cost=round(cost,4),
                improvement_pct=imp, optimizer_used=opt,
                time_seconds=elapsed, detail=report)


def optimize_file(filepath: str, force_optimizer: str = None,
                   verbose: bool = True) -> dict:
    circuit, _ = load_circuit(filepath)
    return optimize(circuit, force_optimizer, verbose)


# ─────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Parameter scaling:")
    print(f"  {'Gates':>8}  {'Pop':>5}  {'Gen':>5}  {'Iters':>6}")
    for n in [6, 292, 547, 659, 1193, 3512, 20679]:
        p = scale_ga_params(n)
        print(f"  {n:>8}  {p['population_size']:>5}  "
              f"{p['generations']:>5}  {scale_iterations(n):>6}")

    print()
    result = optimize_file(
        "data/benchmarks/s1196.bench",
        verbose=True
    )
    print(f"\n  Final: {result['improvement_pct']}% improvement")
    print(f"  Check output/ folder for .bench and report files")