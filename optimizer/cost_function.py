# optimizer/cost_function.py
# PAC Cost Function — Power, Area, Cost estimation
# Based on the math from the paper:
# "Revolutionizing VLSI: AI-Driven Strategies for Power, Area, and Cost Optimization"

# ─────────────────────────────────────────────────────────────
# GATE WEIGHTS (research-based estimates)
# These represent relative cost of each gate type
# in terms of transistor count and switching behavior
# ─────────────────────────────────────────────────────────────

# Power weight = how much switching power this gate consumes
# Based on transistor count and activity factor
GATE_POWER_WEIGHT = {
    'NOT'  : 1.0,   # 2 transistors  - simplest
    'BUFF' : 1.0,   # 2 transistors
    'AND'  : 2.5,   # 6 transistors
    'OR'   : 2.5,   # 6 transistors
    'NAND' : 2.0,   # 4 transistors  - faster than AND
    'NOR'  : 2.0,   # 4 transistors
    'XOR'  : 4.0,   # 12 transistors - complex
    'XNOR' : 4.0,   # 12 transistors
    'DFF'  : 6.0,   # flip flop      - highest power
}

# Area weight = silicon area estimate per gate (in unit cells)
GATE_AREA_WEIGHT = {
    'NOT'  : 1.0,
    'BUFF' : 1.0,
    'AND'  : 1.5,
    'OR'   : 1.5,
    'NAND' : 1.2,
    'NOR'  : 1.2,
    'XOR'  : 2.5,
    'XNOR' : 2.5,
    'DFF'  : 4.0,
}

# Default for unknown gate types
DEFAULT_POWER = 2.0
DEFAULT_AREA  = 1.5

# ─────────────────────────────────────────────────────────────
# COST FUNCTION WEIGHTS
# α, β, γ from the paper
# Tune these to prioritize different objectives
# ─────────────────────────────────────────────────────────────
ALPHA = 0.4   # Power weight
BETA  = 0.4   # Area weight
GAMMA = 0.2   # Wirelength/connectivity weight


# ─────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────

def compute_power(gates):
    """
    Estimates total switching power of the circuit.

    Formula from paper:
        P_switching = A × F × Ceff × Vdd²

    We simplify to:
        power = Σ (gate_power_weight × switching_activity)

    Switching activity estimated as:
        - 1.0 for combinational gates
        - 0.5 for DFFs (toggle rate lower)
    """
    total_power = 0.0

    for out_signal, (gate_type, gate_inputs) in gates.items():
        weight   = GATE_POWER_WEIGHT.get(gate_type, DEFAULT_POWER)
        fan_in   = len(gate_inputs)

        # Switching activity increases with fan_in
        activity = 1.0 if gate_type != 'DFF' else 0.5

        # Power scales with fan_in (more inputs = more transitions)
        gate_power = weight * activity * (1 + 0.1 * fan_in)
        total_power += gate_power

    return round(total_power, 4)


def compute_area(gates):
    """
    Estimates total chip area of the circuit.

    Formula:
        area = Σ (gate_area_weight × complexity_factor)

    Complexity factor accounts for:
        - fan_in (wider gates need more silicon)
        - gate type
    """
    total_area = 0.0

    for out_signal, (gate_type, gate_inputs) in gates.items():
        weight    = GATE_AREA_WEIGHT.get(gate_type, DEFAULT_AREA)
        fan_in    = len(gate_inputs)

        # Area scales slightly with fan_in
        gate_area = weight * (1 + 0.05 * fan_in)
        total_area += gate_area

    return round(total_area, 4)


def compute_wirelength(gates, inputs):
    """
    Estimates total wirelength (connectivity cost).

    In real VLSI: wirelength = HPWL (Half Perimeter Wire Length)
    We estimate it as:
        wirelength = Σ fan_in of each gate

    More connections = longer wires = higher routing cost.
    This is the routing complexity proxy used in research.
    """
    total_connections = 0

    for out_signal, (gate_type, gate_inputs) in gates.items():
        total_connections += len(gate_inputs)

    return float(total_connections)


def compute_pac_cost(gates, inputs, alpha=ALPHA, beta=BETA, gamma=GAMMA):
    """
    Master PAC cost function.

    Formula:
        Cost = α×Power + β×Area + γ×WireLength

    Lower is better.

    Args:
        gates  : dict from parser {out: (gate_type, [inputs])}
        inputs : list of input signals
        alpha  : power weight (default 0.4)
        beta   : area weight  (default 0.4)
        gamma  : wire weight  (default 0.2)

    Returns:
        dict with all cost components + total
    """
    power      = compute_power(gates)
    area       = compute_area(gates)
    wirelength = compute_wirelength(gates, inputs)

    # Normalize wirelength to same scale as power/area
    # Divide by gate count to get per-gate average
    gate_count = max(len(gates), 1)
    wire_norm  = wirelength / gate_count

    total_cost = (alpha * power) + (beta * area) + (gamma * wire_norm)

    return {
        'power'      : power,
        'area'       : area,
        'wirelength' : wirelength,
        'wire_norm'  : round(wire_norm, 4),
        'total_cost' : round(total_cost, 4),
        'alpha'      : alpha,
        'beta'       : beta,
        'gamma'      : gamma,
    }


def print_cost_report(cost_dict, circuit_name="circuit"):
    """Prints a clean cost report."""
    print()
    print("=" * 45)
    print(f"  PAC COST REPORT — {circuit_name}")
    print("=" * 45)
    print(f"  Power      : {cost_dict['power']}")
    print(f"  Area       : {cost_dict['area']}")
    print(f"  WireLength : {cost_dict['wirelength']}")
    print(f"  Wire(norm) : {cost_dict['wire_norm']}")
    print("-" * 45)
    print(f"  Weights    : α={cost_dict['alpha']} "
          f"β={cost_dict['beta']} "
          f"γ={cost_dict['gamma']}")
    print("-" * 45)
    print(f"  TOTAL COST : {cost_dict['total_cost']}")
    print("=" * 45)
    print()


# ── Quick test ───────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
    from netlist_parser import parse_bench

    # Test on c17
    print("Testing PAC cost on c17...")
    inputs, outputs, gates = parse_bench("data/benchmarks/c17.bench")
    cost = compute_pac_cost(gates, inputs)
    print_cost_report(cost, "c17")

    # Test on s38584 (20,000+ gates)
    s38584_path = "data/benchmarks/s38584.bench"
    if os.path.exists(s38584_path):
        print("Testing PAC cost on s38584 (20,679 gates)...")
        inputs2, outputs2, gates2 = parse_bench(s38584_path)
        cost2 = compute_pac_cost(gates2, inputs2)
        print_cost_report(cost2, "s38584")