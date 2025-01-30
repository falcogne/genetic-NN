import random
import copy

from keras import layers
import tensorflow as tf


class TerminalLayer(tf.keras.layers.Layer):

    def __init__(self, vector_rep, vector_choices):
        """
        Take vector rep and choices and create layer

        if vector rep is None, select randomly based on choices
        """
        super(TerminalLayer, self).__init__()

        self.vector_choices = vector_choices

        if vector_rep:
            assert len(vector_rep) == len(vector_choices)
            self.vector_rep = vector_rep
            # If there are none values, fill them in randomly
            for i, ele in enumerate(self.vector_rep):
                if ele is None:
                    self.vector_rep[i] = random.choice(self.vector_choices[i])
        else:
            # randomy fill all values, none were passed in
            self.vector_rep = [random.choice(l) for l in self.vector_choices]

        self.layer = self.create_terminal_layer()


    def create_terminal_layer(self, **kwargs):
        raise NotImplementedError("TerminalLayer subclass needs to define this")


    def call(self, inputs):
        return self.layer(inputs)


    def mutate(self):
        """
        change some structure of this layer by changing the vector representation

        if it decides to change index 1, the layer type will change

        it cannot change the layer from 1d to 2d or vice versa
        """
        ind = random.randint(0, len(self.vector_rep)-1)
        self.vector_rep[ind] = random.choice(self.vector_choices[ind])
        self.layer = self.create_terminal_layer()


class TerminalLayer1D(TerminalLayer):
    """
    Layer within a MalleableLayer1D that actually does the calculation

    Can be any of these options for 1D:
    - Dense
    - Conv1D

    Can be any of these options for 2D:
    - Conv2D
    
    Can be these for either:
    - Dropout
    """

    def __init__(self, vector_rep=None, vector_choices=[range(0, 3), (0.1, 0.3, 0.5, 0.7, 0.9), range(0,257), (1,3,5,7), range(0,3)]):
        """
        define the current layer

        indicies:              [0,                1,           2,        3,             4              ]        
        vector representation: [layer type index, dropout pct, features, kernel size 0, activation func]
        range for each (incl): [(0-2),            (0.0-1.0),   (0-256),  (1,3,5,7),     (0-3?),        ]
        """
        super(TerminalLayer1D, self).__init__(vector_rep, vector_choices)

    
    def create_terminal_layer(self, **kwargs):
        layer_ind, dropout_pct, feature_num, kernel_size_0, activation_func_num = self.vector_rep
        activation_string = ['sigmoid', 'tanh', 'relu'][activation_func_num]

        if layer_ind == 0:
            return layers.Dropout(dropout_pct)
        elif layer_ind == 1:
            return layers.BatchNormalization()
        else:
            # if it's normal data
            return layers.Dense(feature_num, activation=kwargs.get('activation', activation_string), **kwargs)
            # if it's sequence data
            # return layers.Conv1D(feature_num, kernel_size=kwargs.get('kernel_size', kernel_size_0), activation=activation_string, **kwargs)


    def set_unbuilt(self):
        self.built=False
        self.layer.built=False


    def copy(self):
        new_layer = TerminalLayer1D(
            vector_rep=copy.deepcopy(self.vector_rep),
            vector_choices=copy.deepcopy(self.vector_choices),
        )
        return new_layer


    def __str__(self):
        """String representation of the TerminalLayer1D"""
        return f"TerminalLayer1D(type={type(self.layer).__name__}, vector_rep={self.vector_rep})"


class TerminalLayer2D(TerminalLayer):
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
    def __init__(self, vector_rep=None, vector_choices=[range(0, 3), range(1,79), (1,2,3,5,7), (1,2,3,5,7), range(0,3)]):
        """
        define the current layer

        indicies:              [0,                1,        2,             3,             4              ]        
        vector representation: [layer type index, features, kernel size 0, kernel size 1, activation func]
        range for each (incl): [(0-2),            (1-80),  (1,3,5,7),     (1,3,5,7),     (0-3?),        ]
        """
        super(TerminalLayer2D, self).__init__(vector_rep, vector_choices)
        # make the kernel square at the start
        self.vector_rep[3] = self.vector_rep[2]

    
    def create_terminal_layer(self, **kwargs):
        layer_ind, feature_num, kernel_size_0, kernel_size_1, activation_func_num = self.vector_rep
        activation_string = ['sigmoid', 'tanh', 'relu'][activation_func_num]
        if layer_ind == 0:
            return layers.MaxPool2D()
        elif layer_ind == 1:
            return layers.BatchNormalization()
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
