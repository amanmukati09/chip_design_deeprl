# optimizer/logic_rewriter.py
import copy
import random

def get_fanout_counts(gates):
    """Calculates how many times each wire is used as an input."""
    counts = {}
    for _, (_, fanins) in gates.items():
        for f in fanins:
            counts[f] = counts.get(f, 0) + 1
    return counts

def rewrite_redundant_inputs(gates, outputs=None):
    """Rule 1: AND(A, A) -> BUFF(A)"""
    new_gates = copy.deepcopy(gates)
    for out_sig, (gtype, fanins) in list(new_gates.items()):
        if gtype in ['AND', 'OR'] and len(fanins) == 2 and fanins[0] == fanins[1]:
            new_gates[out_sig] = ('BUFF', [fanins[0]])
    return new_gates

def rewrite_double_negation(gates, outputs=None):
    """Rule 2: NOT(NOT(A)) -> BUFF(A)"""
    new_gates = copy.deepcopy(gates)
    not_gates = {out_sig: fanins[0] for out_sig, (gtype, fanins) in new_gates.items() if gtype == 'NOT'}
    
    for out_sig, in_sig in not_gates.items():
        if in_sig in not_gates:
            source_sig = not_gates[in_sig]
            new_gates[out_sig] = ('BUFF', [source_sig])
    return new_gates

def rewrite_absorb_inverters(gates, outputs=None):
    """Rule 3: AND + NOT -> NAND (Strictly checked for fan-out)"""
    new_gates = copy.deepcopy(gates)
    fanouts = get_fanout_counts(new_gates)
    
    not_gates = {out_sig: fanins[0] for out_sig, (gtype, fanins) in new_gates.items() if gtype == 'NOT'}
    
    for not_out, not_in in not_gates.items():
        if not_in in new_gates:
            # ONLY absorb if the preceding gate feeds NOTHING but this inverter
            # AND the preceding gate is not a primary output itself
            if fanouts.get(not_in, 0) == 1 and (outputs is None or not_in not in outputs):
                prev_gtype, prev_fanins = new_gates[not_in]
                if prev_gtype == 'AND':
                    new_gates[not_in] = ('NAND', prev_fanins)
                    new_gates[not_out] = ('BUFF', [not_in])
                elif prev_gtype == 'OR':
                    new_gates[not_in] = ('NOR', prev_fanins)
                    new_gates[not_out] = ('BUFF', [not_in])
    return new_gates

def apply_logic_rewrite(gates, inputs=None, outputs=None):
    strategies = [rewrite_redundant_inputs, rewrite_double_negation, rewrite_absorb_inverters]
    return random.choice(strategies)(gates, outputs)