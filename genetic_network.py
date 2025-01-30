from keras import layers, Model
from keras.layers import Dense
from malleable_skeleton import MalleableLayer

class GeneticNetwork:
    def __init__(self, input_shape, output_features, output_activation_str:str):
        """
        Initialize the GeneticNetwork with a MalleableLayer and output details.

        Args:
            input_shape (tuple): Shape of the input data (excluding batch size).
            output_features (int): Number of output units (e.g., for classification).
            output_activation (str): Activation function for the output layer.
        """

        # TODO: keep the optimization string here and make it malleable??
        self.input_shape = input_shape
        self.output_features = output_features
        self.output_activation_str = output_activation_str

        self.output_layer = Dense(self.output_features, activation=self.output_activation_str)
    

class GeneticNetwork1D(GeneticNetwork):
    def __init__(self, input_shape, output_features, output_activation_str:str="sigmoid"):
        self.malleable_layer = MalleableLayer()
        super(GeneticNetwork1D, self).__init__(input_shape=input_shape, output_features=output_features, output_activation_str=output_activation_str)

    def to_keras_model(self):
        # Define the input layer
        inputs = layers.Input(shape=self.input_shape)

        # Convert the MalleableLayer to a Keras model
        malleable_model = self.malleable_layer.to_keras_model(self.input_shape)
        malleable_output = malleable_model(inputs)

        # Add the output layer based on the specified shape and activation
        outputs = self.output_layer(malleable_output)

        # Create the final model
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def copy(self):
        """
        Create a copy of the GeneticNetwork, including its MalleableLayer.
        
        Returns:
            GeneticNetwork: A deep copy of the network.
        """
        new_network = GeneticNetwork1D(
            input_shape=self.input_shape,
            output_features=self.output_features,
            output_activation_str=self.output_activation_str
        )
        new_network.malleable_layer = self.malleable_layer.copy()
        return new_network
    
    def mutate(self):
        """
        Mutate the internal MalleableLayer structure.
        """
        self.malleable_layer.mutate()

    def __str__(self):
        # # Output details
        # output_details = f"Output shape: {self.output_shape}, Activation: {self.output_activation}"
        
        # Combine all details into a single string
        return (f"GeneticNetwork:\n"
                f"Malleable Layer:\n{str(self.malleable_layer)}\n"
                f"Output:\n{str(self.output_layer)}\n")
                # f"{output_details}")

class GeneticNetwork2D(GeneticNetwork):
    def __init__(self, input_shape, output_features, output_activation_str:str="sigmoid"):
        self.convolution_base = MalleableLayer(is_1d=False)
        self.feedforward = MalleableLayer(is_1d=True)
        super(GeneticNetwork2D, self).__init__(input_shape=input_shape, output_features=output_features, output_activation_str=output_activation_str)

    def to_keras_model(self):
        # Define the input layer
        inputs = layers.Input(shape=self.input_shape)

        # Convert the MalleableLayer to a Keras model
        convolution_model = self.convolution_base.to_keras_model(self.input_shape)
        convolution_output = convolution_model(inputs)
        
        flattened = layers.Flatten()(convolution_output)

        feedforward_model = self.feedforward.to_keras_model(flattened.shape[1:])
        feedforward_output = feedforward_model(flattened)


        # Add the output layer based on the specified shape and activation
        outputs = self.output_layer(feedforward_output)

        # Create the final model
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def copy(self):
        """
        Create a copy of the GeneticNetwork, including its MalleableLayers.
        
        Returns:
            GeneticNetwork: A deep copy of the network.
        """
        new_network = GeneticNetwork2D(
            input_shape=self.input_shape,
            output_features=self.output_features,
            output_activation_str=self.output_activation_str
        )
        new_network.convolution_base = self.convolution_base.copy()
        new_network.feedforward = self.feedforward.copy()
        return new_network
    
    def mutate(self):
        """
        Mutate the internal MalleableLayer structure.
        """
        self.convolution_base.mutate()
        self.feedforward.mutate()
    

    def __str__(self):
        # # Output details
        # output_details = f"Output shape: {self.output_shape}, Activation: {self.output_activation}"
        
        # Combine all details into a single string
        return (f"GeneticNetwork:\n"
                f"Input shape: {self.input_shape}\n"
                f"Convolution base:\n{str(self.convolution_base)}\n"
                f"Feedforward network:\n{str(self.feedforward)}\n"
                f"Output:\n{str(self.output_layer)}\n")
                # f"{output_details}")
