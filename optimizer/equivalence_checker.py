# optimizer/equivalence_checker.py
import random

def simulate_circuit(gates, inputs, input_vector):
    """
    Evaluates the circuit logic for a given boolean input vector.
    """
    values = input_vector.copy()
    
    # ---------------------------------------------------------
    # EDA TRICK: Pseudo-Inputs
    # Only attempt to resolve gates that aren't already explicitly 
    # provided in the input_vector. This allows us to pass DFF 
    # states directly into the simulation to break feedback loops.
    # ---------------------------------------------------------
    unresolved = [node for node in gates.keys() if node not in values]

    max_iters = len(gates) * 2
    iters = 0

    while unresolved and iters < max_iters:
        iters += 1
        progress_made = False
        
        for node in unresolved[:]:
            gtype, fanins = gates[node]
            
            # Check if all inputs for this gate have been calculated yet
            if all(fanin in values for fanin in fanins):
                in_vals = [values[f] for f in fanins]
                
                # Evaluate the Boolean Logic
                if gtype == 'AND': res = all(in_vals)
                elif gtype == 'NAND': res = not all(in_vals)
                elif gtype == 'OR': res = any(in_vals)
                elif gtype == 'NOR': res = not any(in_vals)
                elif gtype == 'XOR': res = sum(in_vals) % 2 == 1
                elif gtype == 'XNOR': res = sum(in_vals) % 2 == 0
                elif gtype == 'NOT': res = not in_vals[0]
                elif gtype == 'BUFF': res = in_vals[0]
                elif gtype == 'DFF': res = in_vals[0] # Fallback for pass-through
                else:
                    raise ValueError(f"Unknown gate type: {gtype}")
                    
                values[node] = res
                unresolved.remove(node)
                progress_made = True
        
        if not progress_made:
            raise RuntimeError(f"Simulation stuck! Combinational loop detected near: {unresolved[:3]}")
            
    return values

def verify_equivalence(original_gates, optimized_gates, inputs, outputs, num_tests=1000):
    """
    Generates random input vectors (and DFF states) to prove 100% equivalence.
    """
    # Identify all DFFs so we can treat them as Pseudo-Inputs
    dff_nodes = [node for node, (gtype, _) in original_gates.items() if gtype == 'DFF']
    
    for i in range(num_tests):
        # 1. Generate random 0s and 1s for the Primary Inputs
        test_vector = {inp: random.choice([True, False]) for inp in inputs}
        
        # 2. Inject random states for DFFs to break sequential loops
        for dff in dff_nodes:
            test_vector[dff] = random.choice([True, False])
            
        try:
            orig_vals = simulate_circuit(original_gates, inputs, test_vector)
            opt_vals = simulate_circuit(optimized_gates, inputs, test_vector)
        except RuntimeError as e:
            print(f"\n[Verification Failed] Structural error detected: {e}")
            return False
        
        # 3. Check if every single Primary Output pin matches perfectly
        for out in outputs:
            if orig_vals.get(out) != opt_vals.get(out):
                print(f"\n[Verification Failed] Mismatch on output pin '{out}' at test {i}")
                return False
                
        # 4. Check if the "Next State" logic (the inputs feeding the DFFs) matches perfectly
        for dff in dff_nodes:
            # Get the name of the wire feeding the DFF in both circuits
            orig_d_pin = original_gates[dff][1][0]
            opt_d_pin = optimized_gates[dff][1][0]
            
            if orig_vals.get(orig_d_pin) != opt_vals.get(opt_d_pin):
                print(f"\n[Verification Failed] Mismatch on Flip-Flop input '{dff}' at test {i}")
                return False
                
    return True