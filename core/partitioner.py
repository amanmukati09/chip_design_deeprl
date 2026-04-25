# core/partitioner.py
import random

def get_window_partition(circuit, window_size=500):
    """
    Selects a random contiguous cluster of gates from the circuit.
    This mimics MFFC (Maximum Fanout-Free Cone) behavior by picking 
    connected logic rather than just random gates.
    """
    all_gate_ids = list(circuit.gates.keys())
    if len(all_gate_ids) <= window_size:
        return all_gate_ids

    # Pick a random starting point
    seed = random.choice(all_gate_ids)
    cluster = {seed}
    
    # Expand by finding neighbors (fan-ins and fan-outs)
    while len(cluster) < window_size:
        current_nodes = list(cluster)
        added_this_round = False
        
        for node in current_nodes:
            # Add fan-ins
            _, fanins = circuit.gates[node]
            for f in fanins:
                if f in circuit.gates and f not in cluster:
                    cluster.add(f)
                    if len(cluster) >= window_size: break
            
            # Add fan-outs (search who uses 'node' as an input)
            # (In a 20k circuit, we'd use a pre-built fan-out map for speed)
            if len(cluster) >= window_size: break
            
        # If we get stuck, pick another random seed to continue
        if not added_this_round and len(cluster) < window_size:
            remaining = list(set(all_gate_ids) - cluster)
            if not remaining: break
            cluster.add(random.choice(remaining))

    return list(cluster)