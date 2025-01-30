import random

from terminal_layers import TerminalLayer, TerminalLayer1D, TerminalLayer2D
from keras import layers, Model

class MalleableLayer():
    def __init__(self, network_order:list=None, sequential:bool=True, is_1d:bool=True):
        """define what's in this node"""
        self.network_order = network_order if network_order is not None else []
        self.is_1d = is_1d
        self.sequential = sequential

        if len(self.network_order) == 0:
            self.network_order.append(self.new_terminal_layer())
        
    def mutate(self):
        raise NotImplementedError("subclass must define mutate")

    def shallow_copy(self, other_malleable_layer):
        """takes the elements of the other layer and puts it into this object, a shallow copy"""
        self.network_order = other_malleable_layer.network_order
        self.is_1d = other_malleable_layer.is_1d

    def new_terminal_layer(self):
        if self.is_1d:
            return TerminalLayer1D()
        else:
            return TerminalLayer2D()
    
    def mutate(self, mutation_selection=None):
        """
        TODO: this comment is wrong
        Mutate the network structure by randomly choosing an operation:
        1. Reorder two layers
        2. Replace a layer
        3. Add a new layer
        4. Switch between sequential and parallel execution
        Args:
            structure (list): The current network structure.
            layer_generator (callable): A function to generate new layers (e.g., Conv2D, Dense, etc.).
        Returns:
            list: The mutated network structure.
        """
        if mutation_selection is None:
            mutation_selection = random.choice(["edit"] * 4 + ["add", "delete"])

        if mutation_selection == "edit":
            mutation_ind = random.randint(0, len(self.network_order)-1)
            to_edit = self.network_order[mutation_ind]
            need_to_del = to_edit.mutate()  # don't propogate the selection so that it can add or delete below
            if need_to_del == "delete the layer above":
                if len(self.network_order) <= 1:
                    return "delete the layer above"
                else:
                    del self.network_order[mutation_ind]

        elif mutation_selection == "add":
            mutation_ind = random.randint(0, len(self.network_order))  # can go full length
            layer = random.choice(['terminal'] * 2 + ["malleable"])
            if layer == "malleable":
                # can only add other type becaues don't want nested structure for no reason
                to_add = MalleableLayer(sequential=not self.sequential, is_1d=self.is_1d)
            elif layer == "terminal":
                to_add = self.new_terminal_layer()

            self.network_order.insert(mutation_ind, to_add)
        
        elif mutation_selection == "delete":
            if len(self.network_order) <= 1:
                return "delete the layer above"  # don't leave an empty structure
            mutation_ind = random.randint(0, len(self.network_order)-1)
            to_delete = self.network_order[mutation_ind]
            if isinstance(to_delete, MalleableLayer) and len(to_delete.network_order) > 1:
                to_delete.mutate(mutation_selection=mutation_selection)
            else:
                del self.network_order[mutation_ind]

        else:
            pass


    def copy(self):
        new_layer = MalleableLayer()
        new_layer.sequential = self.sequential
        new_layer.is_1d = self.is_1d

        new_network_order = []
        for layer in self.network_order:
            new_network_order.append(layer.copy())
        new_layer.network_order = new_network_order
        
        return new_layer

    def __str__(self):
        str_list = [str(ele) for ele in self.network_order]
        
        if self.sequential:
            s = "seq:\n[" + '\n'.join(str_list) + "]"
        else:
            s = f"pll: [{', '.join(str_list)}]"
            
        return s
    

    def to_keras_model(self, input_shape):
        """Convert the structure of this MalleableLayer into a Keras model."""
        inputs = layers.Input(shape=input_shape)

        def process_layer(layer, x):
            """Recursive function to process each layer."""
            if isinstance(layer, MalleableLayer):
                # Recursively process the nested MalleableLayer
                return layer.to_keras_model(x.shape[1:])(x)
            elif isinstance(layer, TerminalLayer):
                # Directly use the Keras layer
                return layer.layer(x)
            elif isinstance(layer, layers.Layer):
                # Directly use the Keras layer
                return layer(x)
            else:
                raise ValueError("Unsupported layer type in network_order")

        if self.sequential:
            x = inputs
            for layer in self.network_order:
                x = process_layer(layer, x)
            outputs = x
        else:
            # Parallel execution: process each layer and merge the outputs
            outputs_list = [process_layer(layer, inputs) for layer in self.network_order]
            outputs = layers.Concatenate()(outputs_list)

        return Model(inputs, outputs)
