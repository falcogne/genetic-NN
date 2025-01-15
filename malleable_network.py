import random

import tensorflow as tf
from keras import layers


LEAF_LAYERS = []


class MalleableLayer(tf.keras.layers.Layer):
    """
    superclass layer that can be altered easily to create a new network structure

    Uses a binary tree structure to store the layer structure. Runs the current node, then the left node, then the right node.

    So to get a sequential layer, put all linear layers in the left subnode, and MalleableLayer as the right subnode.
    Terminate by making the subnodes terminal layers (or to make them blank: None or False)
    """
    def __init__(self, left=False, right=False, sequential=True, mutation_probs=[0.5, 0.5, 1.0, 1.0, 0.06, 0.06]):
        """define the current layer """
        super(MalleableLayer, self).__init__()
        self.left = left
        self.right = right
        self.sequential = sequential
        self.mutation_probs = [sum([mutation_probs[0:i+1]])/100 for i, _ in enumerate(mutation_probs)]

    def call(self, inputs):
        x = inputs
        
        if self.sequential:
            # Sequential mode: run left then right
            x = self.left(x) if self.left else x
            x = self.right(x) if self.right else x
        else:
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
        prob = random.random()
        if prob < self.mutation_probs[0]:
            self.left, self.right = self.right, self.left
        elif prob < self.mutation_probs[1]:
            self.sequential = not self.sequential
        elif prob < self.mutation_probs[2]:
            self.left = random.choice(["TODO: choice of layers to replace with"])
        elif prob < self.mutation_probs[3]:
            self.right = random.choice(["TODO: choice of layers to replace with"])
        elif prob < self.mutation_probs[4]:
            if self.left:
                self.shallow_copy(self.left)  # remove the right node, make the current node the 
        elif prob < self.mutation_probs[4]:
            if self.right:
                self.shallow_copy(self.right)
        else:
            pass  # by default don't change anything; mutations should be rare

        # make the sublayers mutate too we're only calling the mutation from the top
        if self.left is MalleableLayer:
            self.left.mutate()
        if self.right is MalleableLayer:
            self.right.mutate()

    def sex(self, other_layer):
        """combines this layer with another to create a new layer, that is hopefully better than the sum of it's parts"""
        return MalleableLayer(left=self, right=other_layer, sequential=False)
    
    def shallow_copy(self, other_malleable_layer):
        """takes the elements of the other layer and puts it into this object, a shallow copy"""
        self.left = other_malleable_layer.left
        self.right = other_malleable_layer.right
        self.sequential = other_malleable_layer.sequential
        self.mutation_probs = other_malleable_layer.mutation_probs



def example_terminal_layer(units, layer_ind=0, **kwargs):
    if layer_ind == 0:
        return layers.Dense(units, activation=kwargs.get('activation', 'sigmoid'), **kwargs)
    elif layer_ind == 0:
        return layers.Conv1D(units, kernel_size=kwargs.get('kernel_size', 3), activation='relu', **kwargs)
    elif layer_ind == 0:
        return layers.Conv2D(units, kernel_size=kwargs.get('kernel_size', 3), activation='relu', **kwargs)
    elif layer_ind == 0:
        return tf.keras.Sequential([
            layers.Dense(units, kernel_size=kwargs.get('kernel_size', 3), activation='relu', **kwargs)
        ])
    else:
        raise ValueError(f"Unsupported layer index: {layer_ind}")


if __name__ == "__main__":
    linear_layer1 = example_terminal_layer(20)
    linear_layer2 = example_terminal_layer(30)
    linear_layer3 = example_terminal_layer(10)
    linear_layer4 = example_terminal_layer(5)

    # Define restructurable layers
    L4 = MalleableLayer(left=linear_layer4, right=None)
    L3 = MalleableLayer(left=linear_layer3, right=L4)
    L2 = MalleableLayer(left=linear_layer2, right=L3)
    L1 = MalleableLayer(left=linear_layer1, right=L2)

    # Input and model
    model_input = tf.keras.Input(shape=(10,))
    model_output = L1(model_input)

    # Final model
    model = tf.keras.Model(inputs=model_input, outputs=model_output)
    model.summary()