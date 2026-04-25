"""
core/parsers/bench_parser.py
─────────────────────────────────────────────────────
Wrapper around the original netlist_parser.py.

Does not modify the original parser — just exposes
a standard interface: parse(filepath) → (inputs, outputs, gates, name)

This is the same interface all parsers must return so
parser_factory.py can treat them identically.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from core.netlist_parser import parse_bench


def parse(filepath: str):
    """
    Parse a .bench file.

    Returns:
        inputs  : list of input signal names
        outputs : list of output signal names
        gates   : dict {output_signal: (gate_type, [input_signals])}
        name    : circuit name (from filename)
    """
    inputs, outputs, gates = parse_bench(filepath)
    name = os.path.splitext(os.path.basename(filepath))[0]
    return inputs, outputs, gates, name


if __name__ == "__main__":
    inputs, outputs, gates, name = parse("data/benchmarks/s1196.bench")
    print(f"Circuit : {name}")
    print(f"Inputs  : {len(inputs)}")
    print(f"Outputs : {len(outputs)}")
    print(f"Gates   : {len(gates)}")