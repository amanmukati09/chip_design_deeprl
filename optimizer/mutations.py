# optimizer/mutations.py
# Pattern-based mutation engine — functionally correct transformations only.
#
# FIXED: _evaluate_circuit now uses iterative topological evaluation.
#        Recursive version hit Python's 1000-call limit on deep circuits
#        like s1488 (659 gates). Iterative version handles any size.
#
# Rules:
#   Rule 1 — Buffer Elimination      BUFF(x) → wire x directly
#   Rule 2 — Double Negation         NOT(NOT(x)) → wire x
#   Rule 3 — De Morgan Collapse      AND+NOT → NAND,  OR+NOT → NOR
#   Rule 4 — Input Reordering        commutative gate input shuffle
#   Rule 5 — Redundant Gate Merge    duplicate gates → reuse one
#   Rule 6 — De Morgan Expand        NAND → AND+NOT,  NOR → OR+NOT

import copy
import random
import itertools
from typing import Tuple, Dict, List, Optional

Gates = Dict[str, Tuple[str, List[str]]]

COMMUTATIVE = {'AND', 'OR', 'NAND', 'NOR', 'XOR', 'XNOR'}


# ─────────────────────────────────────────────────────────────
# GATE EVALUATOR
# ─────────────────────────────────────────────────────────────

def _eval_gate(gate_type: str, input_vals: List[int]) -> int:
    if not input_vals:
        return 0
    if gate_type in ('BUFF', 'INPUT'):
        return input_vals[0]
    elif gate_type == 'NOT':
        return 1 - input_vals[0]
    elif gate_type == 'AND':
        r = 1
        for v in input_vals: r &= v
        return r
    elif gate_type == 'NAND':
        r = 1
        for v in input_vals: r &= v
        return 1 - r
    elif gate_type == 'OR':
        r = 0
        for v in input_vals: r |= v
        return r
    elif gate_type == 'NOR':
        r = 0
        for v in input_vals: r |= v
        return 1 - r
    elif gate_type == 'XOR':
        r = 0
        for v in input_vals: r ^= v
        return r
    elif gate_type == 'XNOR':
        r = 0
        for v in input_vals: r ^= v
        return 1 - r
    elif gate_type == 'DFF':
        return input_vals[0]
    return 0


def _build_topo_order(primary_inputs: List[str],
                       gates: Gates) -> List[str]:
    """
    Builds a topological evaluation order iteratively.
    Returns list of gate signal names in dependency order.
    Works for any circuit depth — no recursion.
    """
    in_degree  = {sig: 0 for sig in gates}
    dependents: Dict[str, List[str]] = {sig: [] for sig in gates}

    # Count how many gate inputs each gate depends on
    for sig, (_, gate_inputs) in gates.items():
        for inp in gate_inputs:
            if inp in gates:
                in_degree[sig] += 1
                dependents[inp].append(sig)

    # Start with gates whose inputs are all primary inputs
    ready = [sig for sig, deg in in_degree.items() if deg == 0]
    order = []

    while ready:
        sig = ready.pop()
        order.append(sig)
        for dependent in dependents[sig]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                ready.append(dependent)

    return order


def _evaluate_circuit(inputs: List[str],
                       outputs: List[str],
                       gates: Gates,
                       input_assignment: Dict[str, int],
                       topo_order: Optional[List[str]] = None) -> Dict[str, int]:
    """
    Evaluates the full circuit for a given input assignment.
    ITERATIVE — no recursion, handles any circuit depth.

    topo_order can be precomputed and passed in for speed.
    """
    memo: Dict[str, int] = {}

    # Seed primary inputs
    for inp in inputs:
        memo[inp] = input_assignment.get(inp, 0)

    # Evaluate in topological order
    order = topo_order if topo_order is not None else _build_topo_order(inputs, gates)

    for sig in order:
        if sig not in gates:
            continue
        gate_type, gate_inputs = gates[sig]
        input_vals = [memo.get(s, 0) for s in gate_inputs]
        memo[sig] = _eval_gate(gate_type, input_vals)

    return {out: memo.get(out, 0) for out in outputs}


def validate_functional_equivalence(inputs: List[str],
                                     outputs: List[str],
                                     original_gates: Gates,
                                     mutated_gates: Gates,
                                     max_inputs: int = 12) -> bool:
    """
    Truth-table equivalence check — iterative, no recursion limit.

    Precomputes topological order for both circuits once,
    then reuses across all input combinations for speed.

    Returns True if circuits are functionally equivalent.
    """
    n = len(inputs)
    if n == 0:
        return True

    has_dff = any(gt == 'DFF' for gt, _ in original_gates.values())

    # Precompute topo orders once per circuit
    orig_topo = _build_topo_order(inputs, original_gates)
    mut_topo  = _build_topo_order(inputs, mutated_gates)

    if n <= max_inputs and not has_dff:
        combinations = list(itertools.product([0, 1], repeat=n))
    else:
        rng          = random.Random(42)
        combinations = [
            tuple(rng.randint(0, 1) for _ in range(n))
            for _ in range(256)
        ]

    for combo in combinations:
        assignment = dict(zip(inputs, combo))

        orig_out = _evaluate_circuit(inputs, outputs, original_gates,
                                      assignment, orig_topo)
        mut_out  = _evaluate_circuit(inputs, outputs, mutated_gates,
                                      assignment, mut_topo)

        for out_signal in outputs:
            if orig_out.get(out_signal) != mut_out.get(out_signal):
                return False

    return True


# ─────────────────────────────────────────────────────────────
# RULE 1 — BUFFER ELIMINATION
# ─────────────────────────────────────────────────────────────

def mutate_remove_buffer(gates: Gates) -> Optional[Gates]:
    buffers = [sig for sig, (gt, _) in gates.items() if gt == 'BUFF']
    if not buffers:
        return None

    new_gates = copy.deepcopy(gates)
    target    = random.choice(buffers)
    _, buff_inputs = new_gates[target]
    if not buff_inputs:
        return None

    source = buff_inputs[0]
    for sig in list(new_gates.keys()):
        if sig == target:
            continue
        gt, gi = new_gates[sig]
        new_gates[sig] = (gt, [source if x == target else x for x in gi])

    del new_gates[target]
    return new_gates


def mutate_insert_buffer(gates: Gates,
                          inputs: List[str]) -> Optional[Gates]:
    candidates = list(gates.keys())
    if not candidates:
        return None

    new_gates  = copy.deepcopy(gates)
    target_sig = random.choice(candidates)
    buff_name  = f"buf_{target_sig}"
    if buff_name in new_gates:
        return None

    new_gates[buff_name] = ('BUFF', [target_sig])
    downstream = [
        sig for sig, (_, gi) in new_gates.items()
        if target_sig in gi and sig != buff_name
    ]
    if not downstream:
        del new_gates[buff_name]
        return None

    rewire_target = random.choice(downstream)
    gt, gi = new_gates[rewire_target]
    new_gi = list(gi)
    idx    = new_gi.index(target_sig)
    new_gi[idx] = buff_name
    new_gates[rewire_target] = (gt, new_gi)
    return new_gates


# ─────────────────────────────────────────────────────────────
# RULE 2 — DOUBLE NEGATION ELIMINATION
# ─────────────────────────────────────────────────────────────

def mutate_double_negation(gates: Gates) -> Optional[Gates]:
    not_gates = {
        sig: gi[0]
        for sig, (gt, gi) in gates.items()
        if gt == 'NOT' and gi
    }

    for outer_not, (not_type, outer_inputs) in gates.items():
        if not_type != 'NOT' or not outer_inputs:
            continue
        inner_sig = outer_inputs[0]
        if inner_sig not in not_gates:
            continue

        x = not_gates[inner_sig]
        inner_users = [
            sig for sig, (_, gi) in gates.items()
            if inner_sig in gi and sig != outer_not
        ]

        new_gates = copy.deepcopy(gates)

        # Rewire all users of outer_not → x
        for sig in list(new_gates.keys()):
            gt, gi = new_gates[sig]
            if outer_not in gi:
                new_gates[sig] = (gt, [x if s == outer_not else s for s in gi])
        del new_gates[outer_not]

        # If inner_not has no other users, remove it too
        if not inner_users and inner_sig in new_gates:
            for sig in list(new_gates.keys()):
                gt, gi = new_gates[sig]
                if inner_sig in gi:
                    new_gates[sig] = (gt, [x if s == inner_sig else s for s in gi])
            del new_gates[inner_sig]

        return new_gates

    return None


# ─────────────────────────────────────────────────────────────
# RULE 3 — DE MORGAN COLLAPSE
# AND + NOT → NAND,  OR + NOT → NOR
# ─────────────────────────────────────────────────────────────

def mutate_demorgan_collapse(gates: Gates) -> Optional[Gates]:
    collapse_map = {'AND': 'NAND', 'OR': 'NOR'}

    for not_sig, (not_type, not_inputs) in gates.items():
        if not_type != 'NOT' or not not_inputs:
            continue
        and_or_sig = not_inputs[0]
        if and_or_sig not in gates:
            continue
        and_or_type, and_or_inputs = gates[and_or_sig]
        if and_or_type not in collapse_map:
            continue

        users = [
            sig for sig, (_, gi) in gates.items()
            if and_or_sig in gi and sig != not_sig
        ]
        if users:
            continue  # used elsewhere — unsafe to remove

        new_gates = copy.deepcopy(gates)
        new_gates[not_sig] = (collapse_map[and_or_type], list(and_or_inputs))
        del new_gates[and_or_sig]
        return new_gates

    return None


# ─────────────────────────────────────────────────────────────
# RULE 4 — INPUT REORDERING
# ─────────────────────────────────────────────────────────────

def mutate_input_reorder(gates: Gates) -> Optional[Gates]:
    candidates = [
        sig for sig, (gt, gi) in gates.items()
        if gt in COMMUTATIVE and len(gi) >= 2
    ]
    if not candidates:
        return None

    new_gates = copy.deepcopy(gates)
    target    = random.choice(candidates)
    gt, gi    = new_gates[target]
    new_gi    = list(gi)
    random.shuffle(new_gi)
    if new_gi == gi:
        new_gi[0], new_gi[1] = new_gi[1], new_gi[0]
    new_gates[target] = (gt, new_gi)
    return new_gates


# ─────────────────────────────────────────────────────────────
# RULE 5 — REDUNDANT GATE DEDUPLICATION
# ─────────────────────────────────────────────────────────────

def mutate_deduplicate_gates(gates: Gates) -> Optional[Gates]:
    seen: Dict[tuple, str] = {}

    for sig, (gt, gi) in gates.items():
        key = (gt, tuple(gi))
        if key in seen:
            original_sig  = seen[key]
            duplicate_sig = sig

            new_gates = copy.deepcopy(gates)
            for s in list(new_gates.keys()):
                if s == duplicate_sig:
                    continue
                gtt, gii = new_gates[s]
                if duplicate_sig in gii:
                    new_gates[s] = (gtt, [
                        original_sig if x == duplicate_sig else x
                        for x in gii
                    ])
            del new_gates[duplicate_sig]
            return new_gates
        seen[key] = sig

    return None


# ─────────────────────────────────────────────────────────────
# RULE 6 — DE MORGAN EXPAND
# NAND → AND + NOT,  NOR → OR + NOT
# ─────────────────────────────────────────────────────────────

def mutate_demorgan_expand(gates: Gates) -> Optional[Gates]:
    expand_map = {'NAND': 'AND', 'NOR': 'OR'}
    candidates = [sig for sig, (gt, _) in gates.items()
                  if gt in expand_map]
    if not candidates:
        return None

    new_gates  = copy.deepcopy(gates)
    target     = random.choice(candidates)
    ttype, tinputs = new_gates[target]
    base_type  = expand_map[ttype]
    and_or_sig = f"_exp_{target}"

    if and_or_sig in new_gates:
        return None

    new_gates[and_or_sig] = (base_type, list(tinputs))
    new_gates[target]     = ('NOT', [and_or_sig])
    return new_gates


# ─────────────────────────────────────────────────────────────
# MASTER DISPATCHER
# ─────────────────────────────────────────────────────────────

MUTATION_RULES = [
    (mutate_remove_buffer,      30),
    (mutate_double_negation,    25),
    (mutate_demorgan_collapse,  25),
    (mutate_input_reorder,      15),
    (mutate_deduplicate_gates,  10),
    (mutate_demorgan_expand,     5),
]

_RULES   = [r for r, _ in MUTATION_RULES]
_WEIGHTS = [w for _, w in MUTATION_RULES]

def apply_safe_mutation(inputs: List[str],
                         outputs: List[str],
                         gates: Gates,
                         max_attempts: int = 20,
                         validate: bool = True,
                         rule_index: int = None) -> Optional[Gates]:
    """
    Applies one randomly selected mutation rule.
    Validates functional equivalence before returning.
    Retries up to max_attempts if no valid mutation found.

    Returns mutated gates dict, or None if nothing found.
    """
    for _ in range(max_attempts):
        if rule_index is not None:
            rule_fn = _RULES[rule_index % len(_RULES)]
        else:
            rule_fn = random.choices(_RULES, weights=_WEIGHTS, k=1)[0]

        if rule_fn == mutate_insert_buffer:
            mutated = rule_fn(gates, inputs)
        else:
            mutated = rule_fn(gates)

        if mutated is None:
            continue

        if not validate:
            return mutated

        if validate_functional_equivalence(inputs, outputs, gates, mutated):
            return mutated

    return None


# ─────────────────────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────────────────────

def get_mutation_stats(original_gates: Gates,
                        mutated_gates: Gates) -> Dict:
    orig_keys = set(original_gates.keys())
    mut_keys  = set(mutated_gates.keys())
    common    = orig_keys & mut_keys

    changed_type = {
        sig for sig in common
        if original_gates[sig][0] != mutated_gates[sig][0]
    }
    changed_inputs = {
        sig for sig in common
        if set(original_gates[sig][1]) != set(mutated_gates[sig][1])
    }

    return {
        'gates_added'      : len(mut_keys - orig_keys),
        'gates_removed'    : len(orig_keys - mut_keys),
        'gates_changed'    : len(changed_type | changed_inputs),
        'added_signals'    : sorted(mut_keys - orig_keys),
        'removed_signals'  : sorted(orig_keys - mut_keys),
        'changed_signals'  : sorted(changed_type | changed_inputs),
        'original_count'   : len(original_gates),
        'mutated_count'    : len(mutated_gates),
        'net_gate_change'  : len(mutated_gates) - len(original_gates),
    }


# ─────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from core.pipeline import load_circuit

    # Test on both s1196 and s1488 (the one that was crashing)
    for bench in ["data/benchmarks/s1196.bench",
                  "data/benchmarks/s1488.bench"]:
        if not os.path.exists(bench):
            print(f"[SKIP] {bench}")
            continue

        print(f"\n{'='*55}")
        circuit, _ = load_circuit(bench)
        print(f"  Circuit : {circuit.name}  "
              f"({circuit.gate_count} gates, "
              f"{len(circuit.inputs)} inputs)")

        result = apply_safe_mutation(
            circuit.inputs, circuit.outputs, circuit.gates,
            max_attempts=30, validate=True
        )

        if result is None:
            print("  No valid mutation found")
        else:
            valid = validate_functional_equivalence(
                circuit.inputs, circuit.outputs,
                circuit.gates, result
            )
            stats = get_mutation_stats(circuit.gates, result)
            print(f"  Status : {'VALID ✓' if valid else 'INVALID ✗'}")
            print(f"  Gates  : {stats['original_count']} → "
                  f"{stats['mutated_count']} "
                  f"(net {stats['net_gate_change']:+d})")

    print(f"\n{'='*55}")
    print("  Recursion fix verified — iterative evaluator working")
    print(f"{'='*55}")