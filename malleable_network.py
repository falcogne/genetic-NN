import tensorflow as tf
from tensorflow.keras import layers


class MalleableLayer(tf.keras.layers.Layer):
    """
    superclass layer that can be altered easily to create a new network structure

    Uses a binary tree structure to store the layer structure. Runs the current node, then the left node, then the right node.

    So to get a sequential layer, put all linear layers in the left subnode, and MalleableLayer as the right subnode. Terminate by making the right subnode linear (or None)
    """
    def __init__(self, current=None, left=None, right=None, mode='sequential'):
        super(MalleableLayer, self).__init__()
        self.current = current
        self.left = left
        self.right = right
        self.mode = mode

    def call(self, inputs):
        x = self.current(inputs) if self.current else inputs
        
        if self.mode == 'sequential':
            # Sequential mode: left -> right
            x = self.left(x) if self.left else x
            x = self.right(x) if self.right else x
        elif self.mode == 'parallel':
            # Parallel mode: left and right independently, then combine
            left_output = self.left(x) if self.left else x
            right_output = self.right(x) if self.right else x
            x = tf.keras.layers.Add()([left_output, right_output])  # or Concatenate

        return x


def example_terminal_layer(units):
    return tf.keras.Sequential([
        layers.Dense(units, activation='relu')
    ])

if __name__ == "__main__":
    linear_layer1 = example_terminal_layer(20)
    linear_layer2 = example_terminal_layer(30)
    linear_layer3 = example_terminal_layer(10)
    linear_layer4 = example_terminal_layer(5)

    # Define restructurable layers
    L3 = MalleableLayer(current=linear_layer4, left=linear_layer3)
    L2 = MalleableLayer(current=linear_layer2, right=L3)
    L1 = MalleableLayer(current=linear_layer1, right=L2)

    # Input and model
    model_input = tf.keras.Input(shape=(10,))
    model_output = L1(model_input)

    # Final model
    model = tf.keras.Model(inputs=model_input, outputs=model_output)
    model.summary()