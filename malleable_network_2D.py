import random

import tensorflow as tf
from keras import layers

from malleable_network_1D import MalleableLayer1D, TerminalLayer1D

class GeneticNetwork2D(tf.keras.Model):
    """
    Wrapper for MalleableLayer2D that puts together the malleable layer and an output layer
    """
    def __init__(self, input_shape, output_features, output_activation_str='softmax', build=True):
        """define the current layer """
        super(GeneticNetwork2D, self).__init__()
        
        self.orig_input_shape = input_shape
        self.orig_output_features = output_features
        self.orig_output_activation_str = output_activation_str

        self.malleable_convolution_base = MalleableLayer2D()
        self.malleable_feedforward = MalleableLayer1D()
        self.output_layer = layers.Dense(output_features, activation=output_activation_str)
        if build:
            self.build((None,) + self.orig_input_shape)

    # def build(self, input_shape):
    #     self.malleable_layer.build(input_shape)
    #     malleable_output_shape = self.malleable_layer.compute_output_shape(input_shape)
        
    #     self.output_layer.build(malleable_output_shape)

    #     super(GeneticNetwork2D, self).build(input_shape)

    def set_unbuilt(self):
        self.built=False
        self.malleable_convolution_base.set_unbuilt()
        self.malleable_feedforward.set_unbuilt()
        self.output_layer.built=False

    def call(self, inputs):
        x = self.malleable_convolution_base(inputs)
        
        x = tf.keras.layers.Flatten()(x)

        x = self.malleable_feedforward(x)
        x = self.output_layer(x)
        return x

    def force_rebuild(self):
        self.set_unbuilt()
        
        self.malleable_convolution_base.build((None,) + self.orig_input_shape)
        output_shape_conv = self.malleable_convolution_base.compute_output_shape((None,) + self.orig_input_shape)
        flattened_size = int(tf.reduce_prod(output_shape_conv[1:]))

        self.malleable_feedforward.build((None, flattened_size))
        output_shape = self.malleable_feedforward.compute_output_shape((None, flattened_size))
        
        self.output_layer.build(output_shape)

    def mutate(self):
        """
        just mutate the malleable layers
        """
        self.malleable_convolution_base.mutate()
        self.malleable_feedforward.mutate()

    def copy(self):
        """
        Create a copy of the model. Model structure is copied; weights are not

        WILL NEED TO BUILD THE MODEL AFTER THIS COPY
        """
        new_network = GeneticNetwork2D(
            input_shape=self.orig_input_shape,
            output_features=self.orig_output_features,
            output_activation_str=self.orig_output_activation_str,
            build=False
        )

        new_network.malleable_convolution_base = self.malleable_convolution_base.copy()
        new_network.malleable_feedforward = self.malleable_feedforward.copy()
        # new_network.build((None,) + new_network.orig_input_shape)
        return new_network


    def print_structure(self):
        print()
        print("Genetic Network structure:")
        print("CONVOLUTION BASE")
        self.malleable_convolution_base.print_structure()
        print("FEEDFORWARD")
        self.malleable_feedforward.print_structure()
        print(f"Output Layer: Dense(units={self.output_layer.units}, activation={self.output_layer.activation.__name__})")
        print()


class MalleableLayer2D(MalleableLayer1D):
    """
    superclass layer that can be altered easily to create a new network structure

    Uses a binary tree structure to store the layer structure. Runs the current node, then the left node, then the right node.

    So to get a sequential layer, put all linear layers in the left subnode, and MalleableLayer2D as the right subnode.
    Terminate by making the subnodes terminal layers (or to make them blank: None or False)
    """

    @tf.function
    def call(self, inputs):
        x = inputs
        
        if self.sequential:
            # Sequential mode: run left then right
            x = self.left(x) if self.left else x
            x = self.right(x) if self.right else x
        else:
            # Parallel mode: run left and right
            left_output = self.left(x) if self.left else x
            right_output = self.right(x) if self.right else x
            
            # Check if we need to pad the smaller output
            left_shape = tf.shape(left_output)
            right_shape = tf.shape(right_output)
            
            # Compare height and width and decide which one needs padding
            height_diff = left_shape[1] - right_shape[1]
            width_diff = left_shape[2] - right_shape[2]
            
            if height_diff > 0:
                right_output = tf.pad(right_output, 
                                    paddings=[[0, 0],
                                              [height_diff // 2, height_diff - height_diff // 2], 
                                              [0, 0],
                                              [0, 0]],
                                    constant_values=0)
            elif height_diff < 0:
                left_output = tf.pad(left_output, 
                                    paddings=[[0, 0],
                                              [-height_diff // 2, -height_diff - (-height_diff // 2)], 
                                              [0, 0],
                                              [0, 0]],
                                    constant_values=0)
            
            if width_diff > 0:
                right_output = tf.pad(right_output, 
                                    paddings=[[0, 0],
                                              [0, 0],
                                              [width_diff // 2, width_diff - width_diff // 2], 
                                              [0, 0]],
                                    constant_values=0)
            elif width_diff < 0:
                left_output = tf.pad(left_output, 
                                    paddings=[[0, 0],
                                              [0, 0],
                                              [-width_diff // 2, -width_diff - (-width_diff // 2)], 
                                              [0, 0]],
                                    constant_values=0)
            
            # Concatenate the left and right outputs along the channel axis
            x = tf.keras.layers.Concatenate()([left_output, right_output])

        return x

    def build(self, input_shape):
        if self.sequential:
            # Sequential: Left -> Right
            if self.left:
                self.left.build(input_shape)
                left_output_shape = self.left.compute_output_shape(input_shape)
            else:
                left_output_shape = input_shape
            
            if self.right:
                self.right.build(left_output_shape)
            self.built = True
        else:
            # Parallel: Left and Right run on the same input, then concatenated
            left_output_shape = None
            right_output_shape = None

            if self.left:
                self.left.build(input_shape)
                left_output_shape = self.left.compute_output_shape(input_shape)
            
            if self.right:
                self.right.build(input_shape)
                right_output_shape = self.right.compute_output_shape(input_shape)
            
            # Check if left and right output shapes can be concatenated
            if self.left and self.right:
                if left_output_shape[0] != right_output_shape[0]:
                    raise ValueError("Left and Right outputs have incompatible batch sizes for concatenation.")
                
                # Concatenate along channels (depth), adjusting other dimensions if needed
                self.concat_output_shape = (
                    left_output_shape[0],  # Batch size remains the same
                    max(left_output_shape[1], right_output_shape[1]),  # Height
                    max(left_output_shape[2], right_output_shape[2]),  # Width
                    left_output_shape[3] + right_output_shape[3]  # Depth (channels) after concat
                )
            elif self.left:
                # Concatenate left output with input (because right is None)
                self.concat_output_shape = (
                    left_output_shape[0],  # Batch size remains the same
                    max(left_output_shape[1], input_shape[1]),  # Height
                    max(left_output_shape[2], input_shape[2]),  # Width
                    left_output_shape[3] + input_shape[3],  # Concatenate along width (e.g., channels)
                )
            elif self.right:
                # Concatenate right output with input (because left is None)
                self.concat_output_shape = (
                    right_output_shape[0],  # Batch size remains the same
                    max(right_output_shape[1], input_shape[1]),  # Height
                    max(right_output_shape[2], input_shape[2]),  # Width
                    right_output_shape[3] + input_shape[3],  # Concatenate along width (e.g., channels)
                )
            else:
                # If neither left nor right, just double the input channels (fallback)
                self.concat_output_shape = (input_shape[0], input_shape[1], input_shape[2], input_shape[3] * 2)
            
            super(MalleableLayer1D, self).build(input_shape)



    def mutate(self):
        """
        change some structure of this layer. Options are:
        - swap left and right
        - change parallel and sequential ordering
        - change what's in either sublayer
        - only return one sublayer
        - DO NOTHING (most likely outcome, don't want to change everything all the time)

        at the end call mutate on sublayers, we want the mutations to 
        """
        selection = random.randint(0, 10)
        # swap left and right sublayers
        if selection == 0:
            self.left, self.right = self.right, self.left
        # flip sequential/parallel
        elif selection == 1:
            self.sequential = not self.sequential
        # insert/replace a sublayer
        elif selection == 2 or selection == 3:
            if not self.left:
                self.left = random.choice([MalleableLayer2D(), TerminalLayer2D()])
                return
            elif not self.right:
                self.right = random.choice([MalleableLayer2D(), TerminalLayer2D()])
                return
        # remove a sublayer
        elif selection == 4:
            self.left = False
        elif selection == 5:
            self.right = False
        # insert a malleable layer before a sublayer
        elif selection == 6:
            self.left = MalleableLayer2D(left=self.left)
        elif selection == 7:
            self.right = MalleableLayer2D(left=self.right)
        else:
            pass  # don't mutate anything, going to mutate the subtree anyways

        # make the sublayers mutate too we're only calling the mutation from the top
        if isinstance(self.left, MalleableLayer2D):
            self.left.mutate()
        if isinstance(self.right, MalleableLayer2D):
            self.right.mutate()

    def combine(self, other_layer):
        """combines this layer with another to create a new layer, that is hopefully better than the sum of it's parts"""
        return MalleableLayer2D(left=self, right=other_layer, sequential=False)

    def copy(self):
        new_layer = MalleableLayer2D()
        new_layer.sequential = self.sequential
        if self.left:
            new_layer.left = self.left.copy()  # Recursively copy left if it's also a MalleableLayer2D
        if self.right:
            new_layer.right = self.right.copy()  # Recursively copy right if it's also a MalleableLayer2D
        return new_layer

    def print_structure(self, indent=0):
        """Recursively print the structure of the MalleableLayer2D and its sub-layers"""
        indent_str = "  " * indent
        print(f"{indent_str}MalleableLayer2D(sequential={self.sequential})")

        if isinstance(self.left, MalleableLayer2D):
            self.left.print_structure(indent + 1)
        elif isinstance(self.left, TerminalLayer2D):
            print(f"{indent_str}  Left: {self.left}")

        if isinstance(self.right, MalleableLayer2D):
            self.right.print_structure(indent + 1)
        elif isinstance(self.right, TerminalLayer2D):
            print(f"{indent_str}  Right: {self.right}")


class TerminalLayer2D(TerminalLayer1D):
    """
    Layer within a MalleableLayer2D that actually does the calculation

    Can be any of these options for 1D:
    - Dense
    - Conv1D

    Can be any of these options for 2D:
    - Conv2D
    
    Can be these for either:
    - Dropout
    """
    def __init__(self, vector_rep=None, vector_choices=[range(0, 3), range(1,257), (1,2,3,5,7), (1,2,3,5,7), range(0,3)]):
        """
        define the current layer

        indicies:              [0,                1,        2,             3,             4              ]        
        vector representation: [layer type index, features, kernel size 0, kernel size 1, activation func]
        range for each (incl): [(0-2),            (0-256),  (1,3,5,7),     (1,3,5,7),     (0-3?),        ]
        """

        self.vector_choices = vector_choices
        
        if vector_rep is None:
            self.vector_rep = [random.choice(l) for l in self.vector_choices]
        else:
            self.vector_rep = vector_rep
            for i, ele in enumerate(self.vector_rep):
                if ele is None:
                    self.vector_rep[i] = random.choice(self.vector_choices[i])
        
        super(TerminalLayer2D, self).__init__(vector_rep=self.vector_rep, vector_choices=self.vector_choices)

        self.layer = self.create_terminal_layer()

    
    def create_terminal_layer(self, **kwargs):
        layer_ind, feature_num, kernel_size_0, kernel_size_1, activation_func_num = self.vector_rep
        activation_string = ['sigmoid', 'tanh', 'relu'][activation_func_num]
        if layer_ind == 0:
            return layers.MaxPool2D()
        else:
            return layers.Conv2D(feature_num, kernel_size=kwargs.get('kernel_size', (kernel_size_0, kernel_size_1)), activation=activation_string, **kwargs)


    def copy(self):
        new_layer = TerminalLayer2D(
            vector_rep=self.vector_rep,
            vector_choices=self.vector_choices,
        )
        return new_layer

    def __str__(self):
        """String representation of the TerminalLayer2D"""
        return f"TerminalLayer2D(type={type(self.layer).__name__}, vector_rep={self.vector_rep})"
