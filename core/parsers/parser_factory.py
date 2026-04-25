"""
core/parsers/parser_factory.py
─────────────────────────────────────────────────────
Auto-detects circuit file format and routes to the
correct parser. All parsers return the same interface:

    inputs, outputs, gates, name

Supported formats:
    .bench  → ISCAS bench format (original)
    .isc    → ISCAS fault simulation format
    .v      → Structural Verilog (coming next)

Usage:
    from core.parsers.parser_factory import parse_circuit
    inputs, outputs, gates, name = parse_circuit("path/to/file.isc")
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


# Registry of supported formats
# Add new parsers here — nothing else needs to change
SUPPORTED_FORMATS = {
    '.bench' : 'core.parsers.bench_parser',
    '.isc'   : 'core.parsers.isc_parser',
    # '.v'   : 'core.parsers.verilog_parser',   ← uncomment when built
    # '.blif': 'core.parsers.blif_parser',       ← future
}


def get_parser(filepath: str):
    """
    Returns the parser module for the given file extension.
    Raises ValueError for unsupported formats.
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext not in SUPPORTED_FORMATS:
        supported = ', '.join(SUPPORTED_FORMATS.keys())
        raise ValueError(
            f"Unsupported format: '{ext}'\n"
            f"Supported formats: {supported}"
        )

    module_path = SUPPORTED_FORMATS[ext]

    # Dynamic import
    import importlib
    parser_module = importlib.import_module(module_path)
    return parser_module


def parse_circuit(filepath: str):
    """
    Parse any supported circuit file format.

    Args:
        filepath : path to circuit file (.bench, .isc, .v, ...)

    Returns:
        inputs  : list of input signal names
        outputs : list of output signal names
        gates   : dict {signal_name: (gate_type, [input_signals])}
        name    : circuit name

    Raises:
        FileNotFoundError  : if file doesn't exist
        ValueError         : if format is not supported
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Circuit file not found: {filepath}")

    ext    = os.path.splitext(filepath)[1].lower()
    parser = get_parser(filepath)

    inputs, outputs, gates, name = parser.parse(filepath)

    return inputs, outputs, gates, name


def supported_formats():
    """Returns list of supported file extensions."""
    return list(SUPPORTED_FORMATS.keys())


# ─────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    test_files = [
        ("data/benchmarks/s1196.bench", ".bench"),
        ("data/benchmarks/c1355.isc",   ".isc"),
    ]

    print("Parser Factory Test")
    print("=" * 45)
    print(f"Supported: {supported_formats()}")
    print()

    for filepath, fmt in test_files:
        if not os.path.exists(filepath):
            print(f"[SKIP] {filepath} not found")
            continue

        print(f"Testing {fmt}: {os.path.basename(filepath)}")
        try:
            inputs, outputs, gates, name = parse_circuit(filepath)
            print(f"  Circuit : {name}")
            print(f"  Inputs  : {len(inputs)}")
            print(f"  Outputs : {len(outputs)}")
            print(f"  Gates   : {len(gates)}")
            print()
        except Exception as e:
            print(f"  ERROR: {e}")
            print()