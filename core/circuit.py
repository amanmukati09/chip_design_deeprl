# core/circuit.py
# Central Circuit object that holds all circuit data in one place

class Circuit:
    """
    Central data structure for the entire project.
    Every module reads from and writes to this object.
    """

    def __init__(self, name, inputs, outputs, gates, graph):
        # Basic identity
        self.name       = name
        self.inputs     = inputs       # list of input signal names
        self.outputs    = outputs      # list of output signal names
        self.gates      = gates        # dict: {out_signal: (gate_type, [inputs])}
        self.graph      = graph        # NetworkX DiGraph

        # Stats — computed later by feature_extractor
        self.gate_count = len(gates)
        self.depth      = 0            # longest path input → output
        self.fan_in     = {}           # {node: number of inputs}
        self.fan_out    = {}           # {node: number of outputs}

        # Features dict — filled by feature_extractor
        self.features   = {}

        # Cost — filled by cost_function
        self.cost       = None

    def summary(self):
        """Prints a clean summary of the circuit."""
        print("=" * 45)
        print(f"  CIRCUIT  : {self.name}")
        print("=" * 45)
        print(f"  Inputs   : {len(self.inputs)}")
        print(f"  Outputs  : {len(self.outputs)}")
        print(f"  Gates    : {self.gate_count}")
        print(f"  Depth    : {self.depth}")
        print(f"  Cost     : {self.cost}")
        if self.features:
            print(f"  Features :")
            for k, v in self.features.items():
                print(f"    {k:20s} : {v}")
        print("=" * 45)

    def copy(self):
        """
        Returns a deep copy of this Circuit.
        Used by the optimizer to create mutations
        without touching the original.
        """
        import copy
        new_circuit = Circuit(
            name    = self.name + "_copy",
            inputs  = list(self.inputs),
            outputs = list(self.outputs),
            gates   = copy.deepcopy(self.gates),
            graph   = self.graph.copy()
        )
        new_circuit.depth      = self.depth
        new_circuit.fan_in     = dict(self.fan_in)
        new_circuit.fan_out    = dict(self.fan_out)
        new_circuit.features   = dict(self.features)
        new_circuit.cost       = self.cost
        return new_circuit