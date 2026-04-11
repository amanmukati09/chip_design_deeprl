# core/netlist_parser.py
# Production-grade parser for .bench netlist format
# Handles all real MCNC benchmark circuits

import re
import os

# All gate types we support
SUPPORTED_GATES = {
    'AND', 'OR', 'NOT', 'NAND', 'NOR',
    'XOR', 'XNOR', 'BUFF', 'BUF', 'DFF'
}

def parse_bench(filepath):
    """
    Parses a .bench netlist file robustly.

    Handles:
        - Any signal name format (N1, 10GAT, G_17)
        - Spaces or no spaces around =
        - Mixed case gate names
        - Comments anywhere on a line
        - Blank lines
        - BUFF and BUF as same gate
        - Direct assignments (n1 = n2)
        - All MCNC benchmark formats

    Returns:
        inputs  : list of input signal names
        outputs : list of output signal names
        gates   : dict of {out_signal: (gate_type, [input_signals])}

    Raises:
        FileNotFoundError : if file does not exist
        ValueError        : if file has no inputs or outputs
    """

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Circuit file not found: {filepath}")

    inputs  = []
    outputs = []
    gates   = {}

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):

            # ── Clean line ───────────────────────────
            # Remove inline comments
            line = line.split('#')[0]
            # Remove all whitespace around the line
            line = line.strip()
            # Skip empty lines
            if not line:
                continue

            # Uppercase for matching only
            line_upper = line.upper()

            # ── INPUT ────────────────────────────────
            if line_upper.startswith('INPUT'):
                match = re.search(r'\((\S+)\)', line)
                if match:
                    signal = match.group(1).strip()
                    if signal and signal not in inputs:
                        inputs.append(signal)
                continue

            # ── OUTPUT ───────────────────────────────
            if line_upper.startswith('OUTPUT'):
                match = re.search(r'\((\S+)\)', line)
                if match:
                    signal = match.group(1).strip()
                    if signal and signal not in outputs:
                        outputs.append(signal)
                continue

            # ── GATE DEFINITION ──────────────────────
            # Pattern: signal = GATETYPE(in1, in2, ...)
            if '=' in line:
                parts = line.split('=', 1)
                out_signal = parts[0].strip()
                rhs        = parts[1].strip()

                # Check if rhs is a gate call
                gate_match = re.match(r'(\w+)\s*\((.+)\)', rhs)

                if gate_match:
                    gate_type   = gate_match.group(1).upper()
                    inputs_raw  = gate_match.group(2)
                    gate_inputs = [
                        s.strip()
                        for s in inputs_raw.split(',')
                        if s.strip()
                    ]

                    # Normalize BUF → BUFF
                    if gate_type == 'BUF':
                        gate_type = 'BUFF'

                    # Warn about unknown gates but still store
                    if gate_type not in SUPPORTED_GATES:
                        print(f"  [WARN] Line {line_num}: "
                              f"Unknown gate '{gate_type}' — stored anyway")

                    if out_signal:
                        gates[out_signal] = (gate_type, gate_inputs)

                else:
                    # Direct assignment: n1 = n2
                    # Treat as a BUFF gate
                    src_signal = rhs.strip()
                    if src_signal and out_signal:
                        gates[out_signal] = ('BUFF', [src_signal])

    # ── Validate ─────────────────────────────────
    if not inputs:
        raise ValueError(f"No INPUT signals found in: {filepath}")
    if not outputs:
        raise ValueError(f"No OUTPUT signals found in: {filepath}")

    return inputs, outputs, gates


def print_stats(inputs, outputs, gates):
    """Prints clean circuit statistics."""
    print()
    print("=" * 45)
    print("          CIRCUIT PARSE RESULTS")
    print("=" * 45)
    print(f"  Inputs  : {len(inputs)}")
    print(f"  Outputs : {len(outputs)}")
    print(f"  Gates   : {len(gates)}")
    print("-" * 45)

    # Gate type breakdown
    gate_types = {}
    for _, (gtype, _) in gates.items():
        gate_types[gtype] = gate_types.get(gtype, 0) + 1

    print("  Gate Type Breakdown:")
    for gtype, count in sorted(gate_types.items()):
        print(f"    {gtype:10s} : {count}")
    print("=" * 45)
    print()


# ── Quick test ───────────────────────────────────
if __name__ == "__main__":

    # Create real c17 MCNC benchmark circuit for testing
    c17_content = """# c17 ISCAS85 Benchmark Circuit
# 5 inputs, 2 outputs, 6 NAND gates

INPUT(N1)
INPUT(N2)
INPUT(N3)
INPUT(N6)
INPUT(N7)

OUTPUT(N22)
OUTPUT(N23)

N10 = NAND(N1,N3)
N11 = NAND(N3,N6)
N16 = NAND(N2,N11)
N19 = NAND(N11,N7)
N22 = NAND(N10,N16)
N23 = NAND(N16,N19)
"""

    # Write and parse c17
    os.makedirs("data/benchmarks", exist_ok=True)
    c17_path = "data/benchmarks/c17.bench"

    with open(c17_path, 'w') as f:
        f.write(c17_content)

    print("Testing with real MCNC s38584 benchmark...")
    inputs, outputs, gates = parse_bench("data/benchmarks/s38584.bench")
    print_stats(inputs, outputs, gates)

    # Also test our toy circuit still works
    toy_path = "data/benchmarks/test.bench"
    if os.path.exists(toy_path):
        print("Testing with toy circuit...")
        inputs2, outputs2, gates2 = parse_bench(toy_path)
        print_stats(inputs2, outputs2, gates2)