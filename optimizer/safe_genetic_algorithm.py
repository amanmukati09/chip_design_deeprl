# optimizer/safe_genetic_algorithm.py
import random
import copy
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from optimizer.cost_function import compute_pac_cost
from optimizer.safe_simulated_annealing import apply_safe_mutation

# ─────────────────────────────────────────────────────────────
# SAFE GENETIC ALGORITHM CORE
# ─────────────────────────────────────────────────────────────

def init_population(base_gates, inputs, outputs, pop_size):
    """
    Creates an initial population by applying random safe mutations.
    Now passing 'outputs' to ensure we don't break the circuit!
    """
    population = []
    for _ in range(pop_size):
        mutated_gates = copy.deepcopy(base_gates)
        for _ in range(random.randint(1, 3)):
            # UPDATED: Passing outputs
            mutated_gates = apply_safe_mutation(mutated_gates, inputs, outputs)
        population.append(mutated_gates)
    return population

def evaluate_population(population, inputs):
    """Calculates the PAC cost for every circuit in the population."""
    evaluated = []
    for gates in population:
        cost = compute_pac_cost(gates, inputs)['total_cost']
        evaluated.append((gates, cost))
    evaluated.sort(key=lambda x: x[1])
    return evaluated

def tournament_selection(evaluated_pop, tournament_size=3):
    """Picks a few random circuits and returns the best one among them."""
    contenders = random.sample(evaluated_pop, tournament_size)
    contenders.sort(key=lambda x: x[1])
    return contenders[0][0] 

def safe_genetic_algorithm(circuit, pop_size=20, generations=50, mutation_rate=0.8, tournament_size=3, verbose=True):
    """
    Runs an equivalence-preserving Evolutionary Algorithm.
    """
    if verbose:
        print("\n" + "=" * 50)
        print("  SAFE GENETIC ALGORITHM (EVOLUTIONARY STRATEGY)")
        print("=" * 50)

    # 1. Initialize (UPDATED: Passing circuit.outputs)
    population = init_population(circuit.gates, circuit.inputs, circuit.outputs, pop_size)
    
    evaluated_pop = evaluate_population(population, circuit.inputs)
    best_gates = copy.deepcopy(evaluated_pop[0][0])
    best_cost = evaluated_pop[0][1]
    
    history = [(0, best_cost)]
    
    # 2. Evolution Loop
    for gen in range(1, generations + 1):
        next_generation = []
        next_generation.append(copy.deepcopy(best_gates)) # Elitism
        
        while len(next_generation) < pop_size:
            parent = tournament_selection(evaluated_pop, tournament_size)
            child = copy.deepcopy(parent)
            
            if random.random() < mutation_rate:
                for _ in range(random.randint(1, 2)):
                    # UPDATED: Passing circuit.outputs
                    child = apply_safe_mutation(child, circuit.inputs, circuit.outputs)
                    
            next_generation.append(child)
            
        population = next_generation
        evaluated_pop = evaluate_population(population, circuit.inputs)
        
        gen_best_gates, gen_best_cost = evaluated_pop[0]
        if gen_best_cost < best_cost:
            best_gates = copy.deepcopy(gen_best_gates)
            best_cost = gen_best_cost
            
        history.append((gen, best_cost))
        
        if verbose and gen % 10 == 0:
            print(f"  Generation {gen:3d} | Best Cost: {best_cost:.4f}")

    return best_gates, best_cost, history