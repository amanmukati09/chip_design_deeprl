# # optimizer/genetic_algorithm.py
# # Genetic Algorithm optimizer for circuit PAC cost reduction
# #
# # How it works:
# #   Population = multiple circuit variations
# #   Each generation:
# #     1. Evaluate all circuits (PAC cost)
# #     2. Select best performers (survivors)
# #     3. Create new variations (crossover + mutation)
# #     4. Repeat for N generations
# #   Return best circuit found

# import random
# import copy
# import sys
# import os

# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# from optimizer.cost_function        import compute_pac_cost
# from optimizer.simulated_annealing  import apply_random_mutation


# # ─────────────────────────────────────────────────────────────
# # POPULATION FUNCTIONS
# # ─────────────────────────────────────────────────────────────

# def create_individual(gates):
#     """
#     Creates one circuit variant by applying
#     a small number of random mutations.
#     This is one 'individual' in the population.
#     """
#     new_gates = copy.deepcopy(gates)
#     # Apply 1-3 mutations to create variation
#     num_mutations = random.randint(1, 3)
#     for _ in range(num_mutations):
#         new_gates = apply_random_mutation(new_gates)
#     return new_gates


# def create_population(gates, population_size):
#     """
#     Creates initial population of circuit variants.
#     First individual is always the original circuit.
#     """
#     population = [copy.deepcopy(gates)]  # keep original
#     for _ in range(population_size - 1):
#         population.append(create_individual(gates))
#     return population


# def evaluate_population(population, inputs):
#     """
#     Scores every individual in the population.
#     Returns list of (cost, gates) sorted best to worst.
#     """
#     scored = []
#     for gates in population:
#         cost = compute_pac_cost(gates, inputs)['total_cost']
#         scored.append((cost, gates))

#     # Sort by cost ascending (lower = better)
#     scored.sort(key=lambda x: x[0])
#     return scored


# def select_survivors(scored_population, survival_rate=0.3):
#     """
#     Keeps the top performing circuits.
#     survival_rate=0.3 means keep top 30%.
#     """
#     cutoff = max(2, int(len(scored_population) * survival_rate))
#     return [gates for _, gates in scored_population[:cutoff]]


# def crossover(parent1, parent2):
#     """
#     Combines two parent circuits to create a child.

#     Strategy:
#         For each gate, randomly inherit from
#         parent1 or parent2.
#         This mixes good traits from both parents.
#     """
#     child = {}
#     all_keys = set(list(parent1.keys()) + list(parent2.keys()))

#     for key in all_keys:
#         if key in parent1 and key in parent2:
#             # Inherit randomly from either parent
#             if random.random() < 0.5:
#                 child[key] = copy.deepcopy(parent1[key])
#             else:
#                 child[key] = copy.deepcopy(parent2[key])
#         elif key in parent1:
#             child[key] = copy.deepcopy(parent1[key])
#         else:
#             child[key] = copy.deepcopy(parent2[key])

#     return child


# def create_next_generation(survivors, population_size, mutation_rate=0.3):
#     """
#     Creates the next generation from survivors.

#     Strategy:
#         1. Keep survivors (elitism)
#         2. Fill rest with crossover children
#         3. Randomly mutate some children
#     """
#     next_gen = list(survivors)  # elitism — keep best

#     while len(next_gen) < population_size:
#         # Pick two random parents from survivors
#         p1 = random.choice(survivors)
#         p2 = random.choice(survivors)

#         # Create child via crossover
#         child = crossover(p1, p2)

#         # Randomly mutate child
#         if random.random() < mutation_rate:
#             child = apply_random_mutation(child)

#         next_gen.append(child)

#     return next_gen


# # ─────────────────────────────────────────────────────────────
# # GENETIC ALGORITHM CORE
# # ─────────────────────────────────────────────────────────────

# def genetic_algorithm(circuit,
#                       population_size = 20,
#                       generations     = 50,
#                       survival_rate   = 0.3,
#                       mutation_rate   = 0.3,
#                       verbose         = True):
#     """
#     Runs Genetic Algorithm on a Circuit object.

#     Args:
#         circuit         : Circuit object from pipeline
#         population_size : number of circuits per generation
#         generations     : how many generations to evolve
#         survival_rate   : fraction of population that survives
#         mutation_rate   : probability of mutation per child
#         verbose         : print progress

#     Returns:
#         best_gates  : optimized gates dict
#         best_cost   : lowest cost found
#         history     : list of (generation, best_cost, avg_cost)
#     """

#     if verbose:
#         print()
#         print("=" * 50)
#         print("  GENETIC ALGORITHM OPTIMIZER")
#         print("=" * 50)
#         print(f"  Circuit     : {circuit.name}")
#         print(f"  Gates       : {circuit.gate_count}")
#         print(f"  Population  : {population_size}")
#         print(f"  Generations : {generations}")
#         print(f"  Start cost  : {circuit.cost}")
#         print("-" * 50)

#     # ── Initialize ───────────────────────────────────
#     population = create_population(circuit.gates, population_size)
#     best_gates = copy.deepcopy(circuit.gates)
#     best_cost  = circuit.cost
#     history    = []

#     # ── Evolution Loop ───────────────────────────────
#     for gen in range(generations):

#         # Evaluate all individuals
#         scored = evaluate_population(population, circuit.inputs)

#         # Track best
#         gen_best_cost = scored[0][0]
#         gen_avg_cost  = sum(c for c, _ in scored) / len(scored)

#         if gen_best_cost < best_cost:
#             best_cost  = gen_best_cost
#             best_gates = copy.deepcopy(scored[0][1])

#         history.append((gen + 1, round(best_cost, 4),
#                         round(gen_avg_cost, 4)))

#         if verbose and (gen % 10 == 0 or gen == generations - 1):
#             improvement = ((circuit.cost - best_cost)
#                            / circuit.cost * 100)
#             print(f"  Gen {gen+1:3d} | "
#                   f"Best: {best_cost:.4f} | "
#                   f"Avg: {gen_avg_cost:.4f} | "
#                   f"Improvement: {improvement:.2f}%")

#         # Select survivors and breed next generation
#         survivors  = select_survivors(scored, survival_rate)
#         population = create_next_generation(
#             survivors, population_size, mutation_rate
#         )

#     if verbose:
#         improvement = ((circuit.cost - best_cost)
#                        / circuit.cost * 100)
#         print("-" * 50)
#         print(f"  Final best cost : {best_cost}")
#         print(f"  Total improvement: {round(improvement, 2)}%")
#         print("=" * 50)

#     return best_gates, best_cost, history


# # ── Quick test ───────────────────────────────────────
# if __name__ == "__main__":
#     import sys, os
#     sys.path.append(os.path.dirname(os.path.dirname(__file__)))

#     from pipeline_v1 import load_circuit

#     # Test on c17 first
#     print("Testing GA on s1196...")
#     circuit, _ = load_circuit("data/benchmarks/s1196.bench")

#     best_gates, best_cost, history = genetic_algorithm(
#         circuit,
#         population_size = 20,
#         generations     = 50,
#         survival_rate   = 0.3,
#         mutation_rate   = 0.3,
#         verbose         = True
#     )

#     print(f"\nOriginal cost : {circuit.cost}")
#     print(f"Optimized cost: {best_cost}")
#     improvement = ((circuit.cost - best_cost) / circuit.cost) * 100
#     print(f"Improvement   : {round(improvement, 2)}%")

#-------------------------------------------------------------------------------------------------------


# optimizer/genetic_algorithm.py
# Genetic Algorithm — pattern-based correct mutations only.
# All mutations validated via truth-table equivalence check.

import random
import copy
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from optimizer.cost_function import compute_pac_cost
from optimizer.mutations     import apply_safe_mutation


def create_individual(gates, inputs, outputs):
    """Creates one circuit variant via 1-3 validated mutations."""
    new_gates     = copy.deepcopy(gates)
    num_mutations = random.randint(1, 3)
    for _ in range(num_mutations):
        mutated = apply_safe_mutation(
            inputs, outputs, new_gates,
            max_attempts=15, validate=True
        )
        if mutated is not None:
            new_gates = mutated
    return new_gates


def create_population(gates, inputs, outputs, population_size):
    population = [copy.deepcopy(gates)]
    for _ in range(population_size - 1):
        population.append(create_individual(gates, inputs, outputs))
    return population


def evaluate_population(population, inputs):
    scored = []
    for gates in population:
        cost = compute_pac_cost(gates, inputs)['total_cost']
        scored.append((cost, gates))
    scored.sort(key=lambda x: x[0])
    return scored


def select_survivors(scored_population, survival_rate=0.3):
    cutoff = max(2, int(len(scored_population) * survival_rate))
    return [gates for _, gates in scored_population[:cutoff]]


def crossover(parent1, parent2):
    """Per-gate random inheritance from either parent."""
    child    = {}
    all_keys = set(list(parent1.keys()) + list(parent2.keys()))
    for key in all_keys:
        if key in parent1 and key in parent2:
            child[key] = copy.deepcopy(
                parent1[key] if random.random() < 0.5 else parent2[key]
            )
        elif key in parent1:
            child[key] = copy.deepcopy(parent1[key])
        else:
            child[key] = copy.deepcopy(parent2[key])
    return child


def create_next_generation(survivors, inputs, outputs,
                            population_size, mutation_rate=0.3):
    next_gen = list(survivors)
    while len(next_gen) < population_size:
        p1    = random.choice(survivors)
        p2    = random.choice(survivors)
        child = crossover(p1, p2)
        if random.random() < mutation_rate:
            mutated = apply_safe_mutation(
                inputs, outputs, child,
                max_attempts=15, validate=True
            )
            if mutated is not None:
                child = mutated
        next_gen.append(child)
    return next_gen


def genetic_algorithm(circuit,
                       population_size = 20,
                       generations     = 50,
                       survival_rate   = 0.3,
                       mutation_rate   = 0.3,
                       validate        = True,
                       verbose         = True):
    """
    Genetic Algorithm with functionally correct mutations.

    Returns:
        best_gates  : optimized gates dict (guaranteed equivalent)
        best_cost   : lowest PAC cost found
        history     : list of (gen, best_cost, avg_cost)
    """
    if verbose:
        print()
        print("=" * 50)
        print("  GENETIC ALGORITHM  (correct mutations)")
        print("=" * 50)
        print(f"  Circuit     : {circuit.name}")
        print(f"  Gates       : {circuit.gate_count}")
        print(f"  Population  : {population_size}")
        print(f"  Generations : {generations}")
        print(f"  Start cost  : {circuit.cost}")
        print(f"  Validate    : {validate}")
        print("-" * 50)

    population = create_population(
        circuit.gates, circuit.inputs, circuit.outputs, population_size
    )
    best_gates = copy.deepcopy(circuit.gates)
    best_cost  = circuit.cost
    history    = []

    for gen in range(generations):
        scored        = evaluate_population(population, circuit.inputs)
        gen_best_cost = scored[0][0]
        gen_avg_cost  = sum(c for c, _ in scored) / len(scored)

        if gen_best_cost < best_cost:
            best_cost  = gen_best_cost
            best_gates = copy.deepcopy(scored[0][1])

        history.append((gen + 1, round(best_cost, 4),
                         round(gen_avg_cost, 4)))

        if verbose and (gen % 10 == 0 or gen == generations - 1):
            improvement = (circuit.cost - best_cost) / circuit.cost * 100
            print(f"  Gen {gen+1:3d} | Best: {best_cost:.4f} | "
                  f"Avg: {gen_avg_cost:.4f} | "
                  f"Improvement: {improvement:.2f}%")

        survivors  = select_survivors(scored, survival_rate)
        population = create_next_generation(
            survivors, circuit.inputs, circuit.outputs,
            population_size, mutation_rate
        )

    if verbose:
        improvement = (circuit.cost - best_cost) / circuit.cost * 100
        print("-" * 50)
        print(f"  Final best cost  : {best_cost}")
        print(f"  Total improvement: {round(improvement, 2)}%")
        print("=" * 50)

    return best_gates, best_cost, history


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from core.pipeline import load_circuit

    circuit, _ = load_circuit("data/benchmarks/s1196.bench")
    best_gates, best_cost, _ = genetic_algorithm(
        circuit, population_size=20, generations=50,
        validate=True, verbose=True
    )
    print(f"\n  Original : {circuit.cost}")
    print(f"  Best     : {best_cost}")
    print(f"  Gain     : {round((circuit.cost-best_cost)/circuit.cost*100,2)}%")