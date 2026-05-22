# optimizer/safe_simulated_annealing.py
import random
import math
import copy
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from optimizer.cost_function import compute_pac_cost
from optimizer.logic_rewriter import apply_logic_rewrite
# ─────────────────────────────────────────────────────────────
# FUNCTIONALLY CORRECT MUTATION RULES
# ─────────────────────────────────────────────────────────────

def mutate_swap_inputs(gates, inputs):
    """
    Picks a random commutative gate and shuffles its input order.
    Logically safe. Changes graph structure for routing/wirelength exploration.
    """
    commutative_gates = {'AND', 'OR', 'NAND', 'NOR', 'XOR', 'XNOR'}
    new_gates = copy.deepcopy(gates)
    
    valid_candidates = [
        out_sig for out_sig, (gtype, fanins) in new_gates.items() 
        if gtype in commutative_gates and len(fanins) > 1
    ]
    
    if not valid_candidates:
        return new_gates
        
    target = random.choice(valid_candidates)
    gtype, fanins = new_gates[target]
    
    random.shuffle(fanins)
    new_gates[target] = (gtype, fanins)
    return new_gates

def mutate_remove_buffer(gates, inputs, outputs=None):
    """Removes BUFF gates ONLY if they are not primary outputs."""
    if outputs is None: outputs = []
    new_gates = copy.deepcopy(gates)
    
    # Strictly avoid removing buffers that act as output pins
    buffers = [out_sig for out_sig, (gtype, _) in new_gates.items() 
               if gtype == 'BUFF' and out_sig not in outputs]
    
    if not buffers:
        return new_gates
        
    target_buff = random.choice(buffers)
    _, buff_inputs = new_gates[target_buff]
    source_sig = buff_inputs[0] 
    
    del new_gates[target_buff]
    
    for out_sig, (gtype, fanins) in new_gates.items():
        if target_buff in fanins:
            new_fanins = [source_sig if pin == target_buff else pin for pin in fanins]
            new_gates[out_sig] = (gtype, new_fanins)
            
    return new_gates

def apply_safe_mutation(gates, inputs=None, outputs=None):
    if inputs is None: inputs = []
    if outputs is None: outputs = []
        
    if random.random() < 0.70:
        mutations = [mutate_swap_inputs, mutate_remove_buffer]
        # Notice we are passing outputs to mutate_remove_buffer now
        mutation_fn = random.choice(mutations)
        if mutation_fn == mutate_remove_buffer:
            return mutation_fn(gates, inputs, outputs)
        else:
            return mutation_fn(gates, inputs)
    else:
        return apply_logic_rewrite(gates, inputs, outputs)

# ─────────────────────────────────────────────────────────────
# SAFE SIMULATED ANNEALING CORE
# ─────────────────────────────────────────────────────────────

def safe_simulated_annealing(circuit, initial_temp=50.0, cooling_rate=0.95, min_temp=0.1, iterations_per_temp=10, verbose=True):
    """Runs SA using strictly logic-preserving mutations."""
    current_gates = copy.deepcopy(circuit.gates)
    best_gates = copy.deepcopy(current_gates)
    
    current_cost = circuit.cost
    best_cost = current_cost
    
    temp = initial_temp
    history = []
    while temp > min_temp:
        for _ in range(iterations_per_temp):
            # PASS THE OUTPUTS HERE
            new_gates = apply_safe_mutation(current_gates, circuit.inputs, circuit.outputs)
            new_cost_dict = compute_pac_cost(new_gates, circuit.inputs)
            new_cost = new_cost_dict['total_cost']
            
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_gates = copy.deepcopy(new_gates)
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_gates = copy.deepcopy(current_gates)
                    best_cost = current_cost
                    
        history.append((temp, current_cost))
        temp *= cooling_rate
        
    return best_gates, best_cost, history


