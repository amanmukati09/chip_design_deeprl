# core/parsers/verilog_parser.py
# Structural Verilog parser (gate-level netlists from EDA tools).
#
# Supported format:
#   gate_keyword  instance_name  (output, input1, input2, ...);
#   First port in the list is always the output signal.
#
# Supports all gate types seen in MCNC/ITC Verilog:
#   not, buf, and, or, nand, nor, xor, xnor
#   nand2..nand9, and2..and9, nor2..nor9, or2..or9  (wide gates)
#   dff (D flip-flop)
#
# Same return interface as bench_parser and isc_parser:
#   parse(filepath) → (inputs, outputs, gates, name)
#
# Usage:
#   from core.parsers.verilog_parser import parse
#   inputs, outputs, gates, name = parse("path/to/c432.v")

import re
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


# ─────────────────────────────────────────────────────────────
# GATE TYPE NORMALISATION
# Maps Verilog primitive keywords → our internal gate type names
# Handles width suffixes: nand2, nand3, nand4 ... → NAND
# ─────────────────────────────────────────────────────────────

def _normalise_gate(verilog_keyword: str) -> str:
    """
    Converts a Verilog gate primitive keyword to internal gate type.

    Examples:
        not      → NOT
        nand     → NAND
        nand2    → NAND
        nand4    → NAND
        and9     → AND
        buf      → BUFF
        xnor2    → XNOR
        dff      → DFF
    """
    kw = verilog_keyword.lower()

    # Strip trailing digit(s) — width suffix (nand2, and9, etc.)
    base = re.sub(r'\d+$', '', kw)

    mapping = {
        'not'  : 'NOT',
        'buf'  : 'BUFF',
        'buff' : 'BUFF',
        'and'  : 'AND',
        'or'   : 'OR',
        'nand' : 'NAND',
        'nor'  : 'NOR',
        'xor'  : 'XOR',
        'xnor' : 'XNOR',
        'dff'  : 'DFF',
    }

    return mapping.get(base, base.upper())


# ─────────────────────────────────────────────────────────────
# PARSER
# ─────────────────────────────────────────────────────────────

def parse(filepath: str):
    """
    Parse a structural Verilog (.v) file.

    Handles:
        - module/endmodule wrapper
        - input / output / wire declarations (single and list)
        - gate instantiation:  gatetype  instname  (out, in1, in2, ...);
        - // line comments and /* block comments */
        - multi-line port lists
        - wide gates: nand4, and9, etc.
        - output ports that are also primary outputs (no separate wire)

    Returns:
        inputs  : list of input signal names
        outputs : list of output signal names
        gates   : dict {out_signal: (gate_type, [input_signals])}
        name    : module name
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Verilog file not found: {filepath}")

    name    = os.path.splitext(os.path.basename(filepath))[0]
    inputs  = []
    outputs = []
    gates   = {}

    with open(filepath, 'r') as f:
        raw = f.read()

    # ── Strip comments ────────────────────────────────────────
    # Block comments: /* ... */
    raw = re.sub(r'/\*.*?\*/', ' ', raw, flags=re.DOTALL)
    # Line comments: // ...
    raw = re.sub(r'//[^\n]*', ' ', raw)

    # ── Extract module name ───────────────────────────────────
    mod_match = re.search(r'\bmodule\s+(\w+)', raw)
    if mod_match:
        name = mod_match.group(1)

    # ── Tokenise into statements (split on semicolons) ────────
    statements = [s.strip() for s in raw.split(';') if s.strip()]

    for stmt in statements:
        # Collapse internal whitespace / newlines
        stmt = ' '.join(stmt.split())

        # Skip module header and endmodule
        if re.match(r'module\s+', stmt) or stmt == 'endmodule':
            continue

        # ── input declaration ─────────────────────────────────
        if re.match(r'input\b', stmt):
            sigs = _extract_signal_list(stmt)
            for s in sigs:
                if s and s not in inputs:
                    inputs.append(s)
            continue

        # ── output declaration ────────────────────────────────
        if re.match(r'output\b', stmt):
            sigs = _extract_signal_list(stmt)
            for s in sigs:
                if s and s not in outputs:
                    outputs.append(s)
            continue

        # ── wire declaration — skip (signals implicit) ────────
        if re.match(r'wire\b', stmt):
            continue

        # ── gate instantiation ────────────────────────────────
        # Pattern: gatetype  instname  (port1, port2, ...)
        gate_match = re.match(
            r'(\w+)\s+(\w+)\s*\(([^)]+)\)', stmt
        )
        if gate_match:
            gate_kw   = gate_match.group(1)
            # group(2) is instance name — we ignore it
            port_list = gate_match.group(3)

            gate_type = _normalise_gate(gate_kw)

            # Skip non-gate keywords that slip through
            if gate_kw.lower() in ('module', 'endmodule',
                                    'input', 'output', 'wire',
                                    'reg', 'assign'):
                continue

            ports = [p.strip() for p in port_list.split(',')
                     if p.strip()]

            if not ports:
                continue

            # First port = output signal
            out_signal  = ports[0]
            gate_inputs = ports[1:]

            if out_signal:
                gates[out_signal] = (gate_type, gate_inputs)

    # ── Validate ──────────────────────────────────────────────
    if not inputs:
        raise ValueError(f"No input signals found in: {filepath}")
    if not outputs:
        raise ValueError(f"No output signals found in: {filepath}")

    return inputs, outputs, gates, name


def _extract_signal_list(stmt: str) -> list:
    """
    Extracts signal names from an input/output/wire declaration.
    Handles:
        input N1, N2, N3;
        input N1,N4,N8, ...
            N34,N37;           (multi-line, already joined)
    """
    # Remove the keyword prefix
    body = re.sub(r'^(input|output|wire|reg)\b\s*', '', stmt,
                  flags=re.IGNORECASE)
    signals = [s.strip() for s in body.split(',') if s.strip()]
    return signals


# ─────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    test_files = [
        "data/benchmarks/c17.v",
        "data/benchmarks/sqrt.v",
    ]

    for filepath in test_files:
        if not os.path.exists(filepath):
            print(f"[SKIP] {filepath} not found")
            continue

        print(f"\nParsing: {filepath}")
        inputs, outputs, gates, name = parse(filepath)

        print(f"  Module  : {name}")
        print(f"  Inputs  : {len(inputs)}")
        print(f"  Outputs : {len(outputs)}")
        print(f"  Gates   : {len(gates)}")

        # Gate type breakdown
        types = {}
        for _, (gt, _) in gates.items():
            types[gt] = types.get(gt, 0) + 1
        print(f"  Gate types:")
        for gt, cnt in sorted(types.items()):
            print(f"    {gt:<8} : {cnt}")

        # Sample gates
        sample = list(gates.items())[:3]
        print(f"  Sample gates:")
        for sig, (gt, gi) in sample:
            print(f"    {sig} = {gt}({', '.join(gi)})")

        # Sample outputs
        print(f"  Outputs : {outputs}")