# core/graph_builder.py
# Converts parsed netlist into a NetworkX directed graph

import networkx as nx
import matplotlib.pyplot as plt

def build_graph(inputs, outputs, gates):
    """
    Builds a directed graph from parsed netlist.
    Nodes = signals, Edges = signal flow direction
    """
    G = nx.DiGraph()

    # Add input nodes
    for inp in inputs:
        G.add_node(inp, node_type='input')

    # Add output nodes
    for out in outputs:
        G.add_node(out, node_type='output')

    # Add gate nodes and edges
    for out_signal, (gate_type, gate_inputs) in gates.items():
        # Add gate node with its type
        G.add_node(out_signal, node_type='gate', gate_type=gate_type)

        # Add edges from each input signal to this gate output
        for inp_signal in gate_inputs:
            G.add_edge(inp_signal, out_signal)

    return G


def print_graph_stats(G):
    """Prints graph level statistics."""
    print("=" * 40)
    print("        GRAPH STATISTICS")
    print("=" * 40)
    print(f"  Nodes : {G.number_of_nodes()}")
    print(f"  Edges : {G.number_of_edges()}")
    print(f"  Inputs  : {[n for n,d in G.nodes(data=True) if d.get('node_type')=='input']}")
    print(f"  Outputs : {[n for n,d in G.nodes(data=True) if d.get('node_type')=='output']}")
    print(f"  Gates   : {[n for n,d in G.nodes(data=True) if d.get('node_type')=='gate']}")
    print("=" * 40)


def visualize_graph(G, title="Circuit Graph"):
    """Draws the circuit graph."""
    # Color nodes by type
    colors = []
    for node in G.nodes(data=True):
        ntype = node[1].get('node_type', 'gate')
        if ntype == 'input':
            colors.append('lightgreen')
        elif ntype == 'output':
            colors.append('tomato')
        else:
            colors.append('skyblue')

    labels = {}
    for node, data in G.nodes(data=True):
        if data.get('node_type') == 'gate':
            labels[node] = f"{node}\n{data.get('gate_type','')}"
        else:
            labels[node] = node

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 5))
    nx.draw(G, pos, labels=labels, node_color=colors,
            node_size=1500, font_size=8,
            arrows=True, arrowsize=20,
            edge_color='gray')
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ── Quick test ──────────────────────────────
if __name__ == "__main__":
    from netlist_parser import parse_bench, print_stats

    inputs, outputs, gates = parse_bench("data/benchmarks/s832.bench")
    print_stats(inputs, outputs, gates)

    G = build_graph(inputs, outputs, gates)
    print_graph_stats(G)
    visualize_graph(G, title="Test Circuit")