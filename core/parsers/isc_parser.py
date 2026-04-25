"""
core/parsers/isc_parser.py
─────────────────────────────────────────────────────
Parser for the ISC (ISCAS) circuit format.

ISC format structure:
  * comment lines start with *
  Each gate entry is 1-2 lines:

  Line 1:
    <line_num>  <signal_name>  <gate_type>  <fanout>  <fanin>  [fault_tags]

  Line 2 (only if fanin > 0):
    <input_signal_1>  <input_signal_2>  ...

Gate types in ISC:
  inpt  → primary input
  from  → fanout stem (branch of a signal — skip, just an alias)
  and, nand, or, nor, not, buff, xor, xnor, dff → logic gates

Output detection:
  ISC does not explicitly mark outputs.
  A gate is an output if it feeds no other gate in the circuit
  (fanout_count = 0 means nothing uses it internally).
  In practice, the final buff gates at the end of c1355 are outputs.

Returns same interface as bench_parser:
  inputs, outputs, gates, name
"""

import os
import sys
import re

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


# ISC gate type → our internal gate type (matches cost_function.py)
ISC_GATE_MAP = {
    'and'  : 'AND',
    'nand' : 'NAND',
    'or'   : 'OR',
    'nor'  : 'NOR',
    'not'  : 'NOT',
    'buff' : 'BUFF',
    'xor'  : 'XOR',
    'xnor' : 'XNOR',
    'dff'  : 'DFF',
    'inpt' : 'INPUT',
    'from' : 'FROM',   # fanout stem — handled specially
}


def parse(filepath: str):
    """
    Parse an .isc file.

    Returns:
        inputs  : list of input signal names
        outputs : list of output signal names
        gates   : dict {signal_name: (gate_type, [input_signals])}
        name    : circuit name
    """
    name = os.path.splitext(os.path.basename(filepath))[0]

    with open(filepath, 'r') as f:
        raw_lines = f.readlines()

    # ── Pass 1: tokenize into gate records ───────────────────
    # Each gate record = (signal_name, gate_type, fanout, fanin)
    # followed by fanin signal IDs on the next line

    gate_records = []   # list of (signal_name, gate_type, fanout, fanin, [input_ids])
    i = 0

    while i < len(raw_lines):
        line = raw_lines[i].strip()
        i += 1

        # Skip comments and blank lines
        if not line or line.startswith('*'):
            continue

        tokens = line.split()
        if len(tokens) < 3:
            continue

        # ISC line format:
        # line_num  signal_name  gate_type  fanout  fanin  [>sa0] [>sa1]
        # token[0] = line number (ignore)
        # token[1] = signal name
        # token[2] = gate type
        # token[3] = fanout count (optional)
        # token[4] = fanin count (optional)

        signal_name = tokens[1]
        gate_type   = tokens[2].lower()

        try:
            fanout = int(tokens[3]) if len(tokens) > 3 else 0
            fanin  = int(tokens[4]) if len(tokens) > 4 else 0
        except ValueError:
            fanout = 0
            fanin  = 0

        # Read input signals from next line if fanin > 0
        input_ids = []
        if fanin > 0 and i < len(raw_lines):
            next_line = raw_lines[i].strip()
            # Input line is pure numbers (signal IDs), no letters
            if next_line and not next_line.startswith('*'):
                parts = next_line.split()
                # Check it looks like signal IDs (numeric-ish)
                if all(p.lstrip('-').isdigit() or
                       re.match(r'^\d+$', p) for p in parts):
                    input_ids = parts
                    i += 1

        gate_records.append((signal_name, gate_type, fanout, fanin, input_ids))

    # ── Pass 2: build signal ID → signal name mapping ────────
    # ISC uses numeric IDs to reference signals across gates.
    # We need to resolve IDs to signal names.

    # Map: line_number_string → signal_name
    # The line number in col[0] of each raw line is the signal ID
    id_to_name = {}
    for raw in raw_lines:
        raw = raw.strip()
        if not raw or raw.startswith('*'):
            continue
        tokens = raw.split()
        if len(tokens) >= 2:
            id_to_name[tokens[0]] = tokens[1]

    # ── Pass 3: classify and build outputs ───────────────────
    inputs  = []
    gates   = {}          # {signal_name: (gate_type, [input_signal_names])}
    fanout_stems = set()  # 'from' lines — these are aliases, not real gates

    # Track which signals are consumed as inputs by other gates
    consumed = set()

    for signal_name, gate_type, fanout, fanin, input_ids in gate_records:

        if gate_type == 'inpt':
            inputs.append(signal_name)

        elif gate_type == 'from':
            # Fanout stem — it's just an alias for the parent signal.
            # We track it so we can resolve references to it.
            fanout_stems.add(signal_name)
            # Map this stem's name back to the parent signal
            if input_ids:
                parent_id   = input_ids[0]
                parent_name = id_to_name.get(parent_id, parent_id)
                id_to_name[signal_name] = parent_name

        else:
            # Logic gate
            mapped_type = ISC_GATE_MAP.get(gate_type, 'BUFF')

            # Resolve input IDs to signal names
            resolved_inputs = []
            for inp_id in input_ids:
                inp_name = id_to_name.get(inp_id, inp_id)
                # If it's a fanout stem, trace back to the real signal
                # (already handled via id_to_name mapping above)
                resolved_inputs.append(inp_name)
                consumed.add(inp_name)

            gates[signal_name] = (mapped_type, resolved_inputs)
            # ── Pass 4: detect outputs ────────────────────────────────
            # In ISC format, fanout=0 means the signal drives no internal gate
            # These are the primary outputs
            outputs = []
            for signal_name, gate_type, fanout, fanin, input_ids in gate_records:
                if gate_type not in ('inpt', 'from') and fanout == 0:
                    outputs.append(signal_name)
            outputs = sorted(outputs)
    # Clean up: remove any INPUT signals that slipped into gates dict
    for inp in inputs:
        if inp in gates:
            del gates[inp]
 
    return inputs, outputs, gates, name
# ─────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    filepath = sys.argv[1] if len(sys.argv) > 1 else "data/benchmarks/c1355.isc"

    print(f"Parsing: {filepath}")
    inputs, outputs, gates, name = parse(filepath)

    print(f"\nCircuit : {name}")
    print(f"Inputs  : {len(inputs)}")
    print(f"Outputs : {len(outputs)}")
    print(f"Gates   : {len(gates)}")

    if gates:
        sample = list(gates.items())[:3]
        print(f"\nSample gates:")
        for sig, (gtype, inps) in sample:
            print(f"  {sig}: {gtype}({inps})")

    if outputs:
        print(f"\nSample outputs: {outputs[:5]}")