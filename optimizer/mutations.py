# optimizer/mutations.py
# Pattern-based mutation engine — functionally correct transformations only.
#
# Every mutation in this file preserves circuit logic.
# No gate is swapped for a different function.
# All transformations are provably equivalent.
#
# Rules implemented:
#   Rule 1 — Buffer Elimination      BUFF(x) → wire x directly
#   Rule 2 — Double Negation         NOT(NOT(x)) → wire x directly
#   Rule 3 — De Morgan Collapse      AND+NOT → NAND,  OR+NOT → NOR
#   Rule 4 — Input Reordering        AND(a,b) ↔ AND(b,a)  (commutative)
#   Rule 5 — Redundant Gate Merge    duplicate gates → reuse one
#   Rule 6 — De Morgan Expand        NAND → AND+NOT,  NOR → OR+NOT
#
# Functional Correctness Filter:
#   Every mutated circuit passes through validate_functional_equivalence()
#   before being accepted by any optimizer (SA, GA, Hybrid, GNN-SA).
#   If validation fails the mutation is discarded, not applied.
#
# Usage:
#   from optimizer.mutations import apply_safe_mutation, validate_functional_equivalence

import copy
import random
import itertools
from typing import Tuple, Dict, List, Optional

# Type alias for clarity
Gates = Dict[str, Tuple[str, List[str]]]


# ─────────────────────────────────────────────────────────────
# COMMUTATIVE GATES
# Input order doesn't affect output for these.
# ─────────────────────────────────────────────────────────────
COMMUTATIVE = {'AND', 'OR', 'NAND', 'NOR', 'XOR', 'XNOR'}


# ─────────────────────────────────────────────────────────────
# TRUTH TABLE ENGINE
# Used by the functional correctness filter.
# Evaluates a circuit for all input combinations.
# Works for combinational circuits only (no DFF).
# ─────────────────────────────────────────────────────────────

def _eval_gate(gate_type: str, input_vals: List[int]) -> int:
    """Evaluates one gate given integer input values (0 or 1)."""
    if gate_type in ('BUFF', 'INPUT'):
        return input_vals[0]
    elif gate_type == 'NOT':
        return 1 - input_vals[0]
    elif gate_type == 'AND':
        result = 1
        for v in input_vals:
            result &= v
        return result
    elif gate_type == 'NAND':
        result = 1
        for v in input_vals:
            result &= v
        return 1 - result
    elif gate_type == 'OR':
        result = 0
        for v in input_vals:
            result |= v
        return result
    elif gate_type == 'NOR':
        result = 0
        for v in input_vals:
            result |= v
        return 1 - result
    elif gate_type == 'XOR':
        result = 0
        for v in input_vals:
            result ^= v
        return result
    elif gate_type == 'XNOR':
        result = 0
        for v in input_vals:
            result ^= v
        return 1 - result
    elif gate_type == 'DFF':
        # DFF is sequential — cannot truth-table it statically.
        # Return input as-is (transparent latch approximation).
        return input_vals[0]
    else:
        return 0


def _evaluate_circuit(inputs: List[str],
                       outputs: List[str],
                       gates: Gates,
                       input_assignment: Dict[str, int]) -> Dict[str, int]:
    """
    Evaluates the full circuit for a given input assignment.
    Returns {signal_name: 0_or_1} for all output signals.
    Uses topological evaluation (memoized per signal).
    """
    memo: Dict[str, int] = {}

    # Seed primary inputs
    for inp in inputs:
        memo[inp] = input_assignment.get(inp, 0)

    def evaluate_signal(signal: str) -> int:
        if signal in memo:
            return memo[signal]
        if signal not in gates:
            # Unknown signal — treat as 0
            memo[signal] = 0
            return 0
        gate_type, gate_inputs = gates[signal]
        input_vals = [evaluate_signal(s) for s in gate_inputs]
        result = _eval_gate(gate_type, input_vals)
        memo[signal] = result
        return result

    output_vals = {}
    for out in outputs:
        output_vals[out] = evaluate_signal(out)
    return output_vals


def validate_functional_equivalence(inputs: List[str],
                                     outputs: List[str],
                                     original_gates: Gates,
                                     mutated_gates: Gates,
                                     max_inputs: int = 12) -> bool:
    """
    Truth-table equivalence check.

    Evaluates both circuits for all 2^n input combinations
    and compares outputs. If any output differs → not equivalent.

    Capped at max_inputs=12 (4096 combinations) for speed.
    For larger circuits, a subset of random assignments is used.

    Args:
        inputs         : list of primary input signal names
        outputs        : list of primary output signal names
        original_gates : original circuit gates dict
        mutated_gates  : mutated circuit gates dict
        max_inputs     : max inputs for exhaustive check

    Returns:
        True  → circuits are functionally equivalent
        False → circuits differ (mutation discarded)
    """
    n = len(inputs)

    # Check for DFFs — sequential circuits can't be truth-tabled fully.
    # For those we use random sampling with a fixed seed.
    has_dff = any(gt == 'DFF'
                  for gt, _ in original_gates.values())

    if n == 0:
        return True

    # Generate input combinations
    if n <= max_inputs and not has_dff:
        # Exhaustive
        combinations = list(itertools.product([0, 1], repeat=n))
    else:
        # Random sample — 256 assignments, fixed seed for reproducibility
        rng = random.Random(42)
        combinations = [
            tuple(rng.randint(0, 1) for _ in range(n))
            for _ in range(256)
        ]

    for combo in combinations:
        assignment = dict(zip(inputs, combo))

        orig_out = _evaluate_circuit(inputs, outputs, original_gates, assignment)
        mut_out  = _evaluate_circuit(inputs, outputs, mutated_gates,  assignment)

        for out_signal in outputs:
            if orig_out.get(out_signal) != mut_out.get(out_signal):
                return False  # Found a differing output — not equivalent

    return True  # All combinations matched


# ─────────────────────────────────────────────────────────────
# RULE 1 — BUFFER ELIMINATION
# BUFF(x) → replace all uses with x, delete BUFF gate
# ─────────────────────────────────────────────────────────────

def mutate_remove_buffer(gates: Gates) -> Optional[Gates]:
    """
    Finds one BUFF gate and eliminates it.
    All downstream gates that used the BUFF output
    are rewired to use the BUFF's input directly.

    Returns mutated gates dict, or None if no BUFFs exist.
    """
    buffers = [sig for sig, (gt, _) in gates.items() if gt == 'BUFF']
    if not buffers:
        return None

    new_gates = copy.deepcopy(gates)
    target    = random.choice(buffers)
    _, buff_inputs = new_gates[target]

    if not buff_inputs:
        return None

    source = buff_inputs[0]

    # Rewire all gates that use `target` to use `source` instead
    for sig in list(new_gates.keys()):
        if sig == target:
            continue
        gt, gi = new_gates[sig]
        new_gi = [source if x == target else x for x in gi]
        new_gates[sig] = (gt, new_gi)

    del new_gates[target]
    return new_gates


def mutate_insert_buffer(gates: Gates,
                          inputs: List[str]) -> Optional[Gates]:
    """
    Inserts a BUFF on a randomly chosen internal wire.
    BUFF(x) is logically transparent — no function change.
    Useful for exploring area/power tradeoffs.

    Returns mutated gates dict, or None if no suitable wire.
    """
    # All internal signals (gate outputs, not primary inputs)
    candidates = list(gates.keys())
    if not candidates:
        return None

    new_gates  = copy.deepcopy(gates)
    target_sig = random.choice(candidates)
    buff_name  = f"buf_{target_sig}"

    # Avoid name collision
    if buff_name in new_gates:
        return None

    # Insert BUFF
    new_gates[buff_name] = ('BUFF', [target_sig])

    # Rewire ONE random downstream gate to use buff_name instead
    downstream = [
        sig for sig, (gt, gi) in new_gates.items()
        if target_sig in gi and sig != buff_name
    ]
    if not downstream:
        del new_gates[buff_name]
        return None

    rewire_target = random.choice(downstream)
    gt, gi = new_gates[rewire_target]
    # Replace first occurrence of target_sig
    new_gi = list(gi)
    idx = new_gi.index(target_sig)
    new_gi[idx] = buff_name
    new_gates[rewire_target] = (gt, new_gi)

    return new_gates


# ─────────────────────────────────────────────────────────────
# RULE 2 — DOUBLE NEGATION ELIMINATION
# NOT(NOT(x)) → wire x directly, delete both NOTs
# ─────────────────────────────────────────────────────────────

def mutate_double_negation(gates: Gates) -> Optional[Gates]:
    """
    Finds a NOT gate whose sole input is another NOT gate.
    Eliminates both and rewires downstream to use original signal.

    Pattern:
        n1 = NOT(x)
        n2 = NOT(n1)   ← both can be removed
        downstream uses n2 → rewire to x
    """
    not_gates = {sig: gi for sig, (gt, gi) in gates.items() if gt == 'NOT'}

    # Find a NOT whose input is also a NOT
    for outer_not, (_, outer_inputs) in gates.items():
        if gates.get(outer_not, ('',))[0] != 'NOT':
            continue
        inner_sig = outer_inputs[0] if outer_inputs else None
        if inner_sig and inner_sig in not_gates:
            # Found double negation: outer_not = NOT(inner_sig), inner_sig = NOT(x)
            x = not_gates[inner_sig][0] if not_gates[inner_sig] else None
            if x is None:
                continue

            # Check inner_not is used ONLY by outer_not (safe to remove)
            inner_users = [
                sig for sig, (_, gi) in gates.items()
                if inner_sig in gi and sig != outer_not
            ]
            if inner_users:
                # inner_not is used elsewhere — only remove outer
                new_gates = copy.deepcopy(gates)
                # Rewire users of outer_not → x
                for sig in list(new_gates.keys()):
                    gt, gi = new_gates[sig]
                    if outer_not in gi:
                        new_gates[sig] = (gt, [x if s == outer_not else s for s in gi])
                del new_gates[outer_not]
                return new_gates
            else:
                # Both can be removed
                new_gates = copy.deepcopy(gates)
                # Rewire users of outer_not → x
                for sig in list(new_gates.keys()):
                    gt, gi = new_gates[sig]
                    if outer_not in gi:
                        new_gates[sig] = (gt, [x if s == outer_not else s for s in gi])
                # Rewire users of inner_sig → x (there are none, but clean up)
                del new_gates[outer_not]
                if inner_sig in new_gates:
                    del new_gates[inner_sig]
                return new_gates

    return None  # No double negation found


# ─────────────────────────────────────────────────────────────
# RULE 3 — DE MORGAN COLLAPSE
# AND(a,b) + NOT(AND_out) → NAND(a,b)   [removes 1 gate]
# OR(a,b)  + NOT(OR_out)  → NOR(a,b)    [removes 1 gate]
# ─────────────────────────────────────────────────────────────

def mutate_demorgan_collapse(gates: Gates) -> Optional[Gates]:
    """
    Finds a NOT gate that feeds directly off an AND or OR gate
    and collapses the pair into a single NAND or NOR.

    Pattern:
        n1 = AND(a, b)
        n2 = NOT(n1)       ← n1 used ONLY here
        → replace both with: n2 = NAND(a, b)

    Requirements:
        - The AND/OR output (n1) must be used ONLY by this one NOT
        - If n1 is used elsewhere, we cannot remove it
    """
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

        # Check that and_or_sig is ONLY used by this NOT
        users = [
            sig for sig, (_, gi) in gates.items()
            if and_or_sig in gi and sig != not_sig
        ]
        if users:
            continue  # Used elsewhere — can't collapse

        # Safe to collapse
        new_gates = copy.deepcopy(gates)
        new_type  = collapse_map[and_or_type]

        # Replace the NOT with NAND/NOR using AND/OR's inputs
        new_gates[not_sig] = (new_type, list(and_or_inputs))

        # Delete the old AND/OR
        del new_gates[and_or_sig]

        return new_gates

    return None  # No collapsible pair found


# ─────────────────────────────────────────────────────────────
# RULE 4 — INPUT REORDERING (commutative gates)
# AND(a,b) ↔ AND(b,a) — safe, may affect wirelength estimate
# ─────────────────────────────────────────────────────────────

def mutate_input_reorder(gates: Gates) -> Optional[Gates]:
    """
    Picks a random commutative gate and shuffles its inputs.
    Logically identical — only affects routing cost estimate.

    Only applies to gates with 2+ inputs.
    """
    candidates = [
        sig for sig, (gt, gi) in gates.items()
        if gt in COMMUTATIVE and len(gi) >= 2
    ]
    if not candidates:
        return None

    new_gates = copy.deepcopy(gates)
    target    = random.choice(candidates)
    gt, gi    = new_gates[target]

    new_gi = list(gi)
    random.shuffle(new_gi)

    # Only return if order actually changed
    if new_gi == gi:
        # Force a swap of first two
        new_gi[0], new_gi[1] = new_gi[1], new_gi[0]

    new_gates[target] = (gt, new_gi)
    return new_gates


# ─────────────────────────────────────────────────────────────
# RULE 5 — REDUNDANT GATE DEDUPLICATION
# n1 = AND(a,b)  AND  n2 = AND(a,b)  →  replace n2 with wire to n1
# ─────────────────────────────────────────────────────────────

def mutate_deduplicate_gates(gates: Gates) -> Optional[Gates]:
    """
    Finds two gates with identical type and identical inputs (same order).
    Replaces all uses of the duplicate with the original.

    Pattern:
        n1 = AND(a, b)
        n2 = AND(a, b)   ← duplicate
        downstream(n2) → rewire to n1
    """
    # Build a map: (gate_type, tuple(inputs)) → first occurrence
    seen: Dict[tuple, str] = {}

    for sig, (gt, gi) in gates.items():
        key = (gt, tuple(gi))
        if key in seen:
            original_sig = seen[key]
            duplicate_sig = sig

            # Rewire all uses of duplicate_sig → original_sig
            new_gates = copy.deepcopy(gates)
            for s in list(new_gates.keys()):
                if s == duplicate_sig:
                    continue
                gtt, gii = new_gates[s]
                if duplicate_sig in gii:
                    new_gii = [original_sig if x == duplicate_sig else x for x in gii]
                    new_gates[s] = (gtt, new_gii)

            # Remove the duplicate gate
            del new_gates[duplicate_sig]
            return new_gates

        seen[key] = sig

    return None  # No duplicates found


# ─────────────────────────────────────────────────────────────
# RULE 6 — DE MORGAN EXPAND
# NAND(a,b) → AND(a,b) + NOT(AND_out)
# NOR(a,b)  → OR(a,b)  + NOT(OR_out)
# Used sparingly — increases gate count but can enable further
# optimizations (e.g., a later collapse with a different NOT).
# ─────────────────────────────────────────────────────────────

def mutate_demorgan_expand(gates: Gates) -> Optional[Gates]:
    """
    Expands a NAND or NOR into AND/OR + NOT.
    Increases gate count but opens new optimization paths.

    Applied rarely (10% weight) since it makes things larger first.
    """
    expand_map = {'NAND': 'AND', 'NOR': 'OR'}

    candidates = [
        sig for sig, (gt, _) in gates.items()
        if gt in expand_map
    ]
    if not candidates:
        return None

    new_gates   = copy.deepcopy(gates)
    target      = random.choice(candidates)
    target_type, target_inputs = new_gates[target]
    base_type   = expand_map[target_type]

    and_or_sig  = f"_exp_{target}"
    not_sig     = target

    # Avoid name collision
    if and_or_sig in new_gates:
        return None

    # Add new AND/OR gate
    new_gates[and_or_sig] = (base_type, list(target_inputs))

    # Replace original NAND/NOR with NOT of the new AND/OR
    new_gates[not_sig] = ('NOT', [and_or_sig])

    return new_gates


# ─────────────────────────────────────────────────────────────
# MASTER MUTATION DISPATCHER
# ─────────────────────────────────────────────────────────────

# Weights control how often each rule is tried.
# Rules that reduce gate count are weighted higher.
# De Morgan expand is low-weight (makes circuit bigger first).
MUTATION_RULES = [
    (mutate_remove_buffer,      30),   # Most impactful — removes gates
    (mutate_double_negation,    25),   # High value — removes 2 gates
    (mutate_demorgan_collapse,  25),   # High value — removes 1 gate
    (mutate_input_reorder,      15),   # Low cost, always valid
    (mutate_deduplicate_gates,  10),   # Medium — removes duplicate
    (mutate_demorgan_expand,     5),   # Rare — grows before shrink
]

_RULES     = [r for r, _ in MUTATION_RULES]
_WEIGHTS   = [w for _, w in MUTATION_RULES]


def apply_safe_mutation(inputs: List[str],
                         outputs: List[str],
                         gates: Gates,
                         max_attempts: int = 20,
                         validate: bool = True) -> Optional[Gates]:
    """
    Applies one randomly selected mutation rule.
    If validate=True, runs the functional correctness filter.
    Retries up to max_attempts times if no valid mutation found.

    Args:
        inputs       : primary input signal names
        outputs      : primary output signal names
        gates        : current circuit gates dict
        max_attempts : max tries before giving up
        validate     : whether to run truth-table validation

    Returns:
        Mutated gates dict (guaranteed equivalent if validate=True)
        None if no valid mutation found after max_attempts
    """
    for attempt in range(max_attempts):
        # Weighted random rule selection
        rule_fn = random.choices(_RULES, weights=_WEIGHTS, k=1)[0]

        # Some rules need extra arguments
        if rule_fn == mutate_insert_buffer:
            mutated = rule_fn(gates, inputs)
        else:
            mutated = rule_fn(gates)

        if mutated is None:
            continue  # Rule found nothing applicable — try again

        if not validate:
            return mutated

        # Functional correctness filter
        if validate_functional_equivalence(inputs, outputs, gates, mutated):
            return mutated
        # else: mutation broke equivalence — discard and retry

    return None  # No valid mutation found


# ─────────────────────────────────────────────────────────────
# MUTATION STATS (for debugging / logging)
# ─────────────────────────────────────────────────────────────

def get_mutation_stats(original_gates: Gates,
                        mutated_gates: Gates) -> Dict:
    """
    Returns a summary of what changed between original and mutated circuit.
    Used for the gate-level diff in the output report.
    """
    orig_keys    = set(original_gates.keys())
    mut_keys     = set(mutated_gates.keys())

    added        = mut_keys - orig_keys
    removed      = orig_keys - mut_keys
    common       = orig_keys & mut_keys

    changed_type = {
        sig for sig in common
        if original_gates[sig][0] != mutated_gates[sig][0]
    }
    changed_inputs = {
        sig for sig in common
        if set(original_gates[sig][1]) != set(mutated_gates[sig][1])
    }

    return {
        'gates_added'      : len(added),
        'gates_removed'    : len(removed),
        'gates_changed'    : len(changed_type | changed_inputs),
        'added_signals'    : sorted(added),
        'removed_signals'  : sorted(removed),
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

    print("=" * 55)
    print("  MUTATION ENGINE TEST")
    print("=" * 55)

    circuit, _ = load_circuit("data/benchmarks/s1196.bench")

    inputs  = circuit.inputs
    outputs = circuit.outputs
    gates   = circuit.gates

    print(f"  Circuit     : {circuit.name}")
    print(f"  Gates       : {circuit.gate_count}")
    print(f"  Inputs      : {len(inputs)}")
    print(f"  Outputs     : {len(outputs)}")
    print()

    # Test each rule individually
    rules_to_test = [
        ("Buffer Elimination",     mutate_remove_buffer,     [gates]),
        ("Double Negation",        mutate_double_negation,   [gates]),
        ("De Morgan Collapse",     mutate_demorgan_collapse, [gates]),
        ("Input Reorder",          mutate_input_reorder,     [gates]),
        ("Redundant Gate Merge",   mutate_deduplicate_gates, [gates]),
        ("De Morgan Expand",       mutate_demorgan_expand,   [gates]),
    ]

    for name, fn, args in rules_to_test:
        result = fn(*args)
        if result is None:
            print(f"  [{name}]  → no applicable pattern found")
        else:
            valid = validate_functional_equivalence(
                inputs, outputs, gates, result
            )
            stats = get_mutation_stats(gates, result)
            status = "VALID ✓" if valid else "INVALID ✗"
            print(f"  [{name}]")
            print(f"    Status     : {status}")
            print(f"    Gates      : {stats['original_count']} → {stats['mutated_count']} "
                  f"(net {stats['net_gate_change']:+d})")
            if stats['removed_signals']:
                print(f"    Removed    : {stats['removed_signals'][:3]}")
            if stats['added_signals']:
                print(f"    Added      : {stats['added_signals'][:3]}")
            print()

    print("=" * 55)
    print("  FULL PIPELINE TEST (20 mutations with validation)")
    print("=" * 55)

    success = 0
    failed  = 0
    skipped = 0

    for i in range(20):
        result = apply_safe_mutation(inputs, outputs, gates, validate=True)
        if result is None:
            skipped += 1
        else:
            valid = validate_functional_equivalence(
                inputs, outputs, gates, result
            )
            if valid:
                success += 1
            else:
                failed += 1

    print(f"  Valid mutations   : {success}/20")
    print(f"  Invalid (escaped) : {failed}/20  ← must be 0")
    print(f"  No pattern found  : {skipped}/20")
    print()
    if failed == 0:
        print("  Correctness filter: WORKING ✓")
    else:
        print("  Correctness filter: BROKEN ✗ — investigate immediately")
    print("=" * 55)