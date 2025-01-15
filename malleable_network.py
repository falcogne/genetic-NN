import random

import tensorflow as tf
from keras import layers


class GeneticNetwork(tf.keras.layers.Layer):
    """
    Wrapper for MalleableLayer that puts together the malleable layer and an output layer
    """
    def __init__(self, input_shape, output_features, output_activation_str='categorical_crossentropy'):
        """define the current layer """
        super(GeneticNetwork, self).__init__()
        self.input_layer = tf.keras.Input(shape=input_shape)
        self.malleable_layer = MalleableLayer()
        self.output_layer = layers.Dense(output_features, activation=output_activation_str)

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.malleable_layer(x)
        x = self.output_layer(x)
        return x

    def mutate(self):
        """
        just mutate the malleable layer
        """
        self.malleable_layer.mutate()


class MalleableLayer(tf.keras.layers.Layer):
    """
    superclass layer that can be altered easily to create a new network structure

    Uses a binary tree structure to store the layer structure. Runs the current node, then the left node, then the right node.

    So to get a sequential layer, put all linear layers in the left subnode, and MalleableLayer as the right subnode.
    Terminate by making the subnodes terminal layers (or to make them blank: None or False)
    """
    def __init__(self, left=False, right=False, sequential=True):
        """define the current layer """
        super(MalleableLayer, self).__init__()
        self.left = left
        self.right = right
        self.sequential = sequential

    def call(self, inputs):
        x = inputs
        
        if self.sequential:
            # Sequential mode: run left then right
            x = self.left(x) if self.left else x
            x = self.right(x) if self.right else x
        else:
            if not self.left and not self.right:
                return x
            # Parallel mode: run left and right and then combine
            left_output = self.left(x) if self.left else x
            right_output = self.right(x) if self.right else x
            x = tf.keras.layers.Add()([left_output, right_output])  # or Concatenate

        return x

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
        if selection == 0:
            self.left, self.right = self.right, self.left
        elif selection == 1:
            self.sequential = not self.sequential
        elif selection == 2 or selection == 3:
            if not self.left:
                self.left = random.choice([MalleableLayer(), TerminalLayer(force_dimension=1)])
                return
            elif not self.right:
                self.right = random.choice([MalleableLayer(), TerminalLayer(force_dimension=1)])
                return
        elif selection == 4:
            if self.left is MalleableLayer:
                self.shallow_copy(self.left)  # remove the right node
                return
        elif selection == 5:
            if self.right is MalleableLayer:
                self.shallow_copy(self.right)  # remove the left node
                return
        else:
            pass  # by default don't change anything; mutations should be rare

        # make the sublayers mutate too we're only calling the mutation from the top
        if self.left is MalleableLayer:
            self.left.mutate()
        if self.right is MalleableLayer:
            self.right.mutate()

    def combine(self, other_layer):
        """combines this layer with another to create a new layer, that is hopefully better than the sum of it's parts"""
        return MalleableLayer(left=self, right=other_layer, sequential=False)
    
    def shallow_copy(self, other_malleable_layer):
        """takes the elements of the other layer and puts it into this object, a shallow copy"""
        self.left = other_malleable_layer.left
        self.right = other_malleable_layer.right
        self.sequential = other_malleable_layer.sequential


class TerminalLayer(tf.keras.layers.Layer):
    """
    Layer within a MalleableLayer that actually does the calculation

    Can be any of these options for 1D:
    - Dense
    - Conv1D

    Can be any of these options for 2D:
    - Conv2D
    
    Can be these for either:
    - Dropout
    """
    def __init__(self, force_dimension=0, vector_rep=[0, 0.5, 1, 128, 3, 3, 0], vector_choices=[range(0,3), (0.1, 0.3, 0.5, 0.7, 0.9), range(1,2), range(0,257), (1,3,5,7), (1,3,5,7), range(0,4)]):
        """
        define the current layer

        indicies:              [0,                  1,                2,           3,        4,             5,             6              ]        
        vector representation: [is one dimensional, layer type index, dropout pct, features, kernel size 0, kernel size 1, activation func]
        range for each (incl): [(0-1),              (0-2),            (0.0-1.0),   (0-256),  (1,3,5,7),     (1,3,5,7),     (0-3?),        ]
        """
        super(TerminalLayer, self).__init__()
        
        self.vector_choices = vector_choices
        # self.vector_rep = vector_rep
        self.vector_rep = [random.choice(l) for l in self.vector_choices]

        if force_dimension in (1,2):
            self.vector_rep[2] = force_dimension

        self.layer = self.create_terminal_layer()

    
    def create_terminal_layer(self, **kwargs):
        is_1d, layer_ind, dropout_pct, feature_num, kernel_size_0, kernel_size_1, activation_func_num = self.vector_rep
        activation_string = ['sigmoid', 'tanh', 'relu'][activation_func_num]
        if is_1d == 1:
            if layer_ind == 0:
                return layers.Dropout(dropout_pct)
            if layer_ind == 1:
                return layers.Conv1D(feature_num, kernel_size=kwargs.get('kernel_size', kernel_size_0), activation=activation_string, **kwargs)
            else:
                return layers.Dense(feature_num, activation=kwargs.get('activation', activation_string), **kwargs)
        else:
            if layer_ind == 0:
                return layers.MaxPool2D()
            else:
                return layers.Conv2D(feature_num, kernel_size=kwargs.get('kernel_size', (kernel_size_0, kernel_size_1)), activation=activation_string, **kwargs)


    def call(self, inputs):
        return self.layer(inputs)


    def mutate(self):
        """
        change some structure of this layer by changing the vector representation

        if it decides to change index 1, the layer type will change

        it cannot change the layer from 1d to 2d or vice versa
        """
        ind = random.randint(1, len(self.vector_rep)-1)
        self.vector_rep[ind] = random.choice(self.vector_choices[ind])
        self.layer = self.create_terminal_layer()

    def combine(self, other_layer):
        """combines this layer with another to create a new layer, that is hopefully better than the sum of it's parts"""
        raise NotImplementedError("shouldn't be using combine I don't think")
        return MalleableLayer(left=self, right=other_layer, sequential=False)
    
    def shallow_copy(self, other_malleable_layer):
        """takes the elements of the other layer and puts it into this object, a shallow copy"""
        self.left = other_malleable_layer.left
        self.right = other_malleable_layer.right
        self.sequential = other_malleable_layer.sequential
        self.mutation_probs = other_malleable_layer.mutation_probs



if __name__ == "__main__":
    starter_model = GeneticNetwork(input_shape=(10,), output_features=1, activation_str='sigmoid')
    
    # Input and model
    # model_output = starter_model(model_input)

    starter_model.summary()