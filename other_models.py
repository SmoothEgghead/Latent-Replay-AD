import torch
from torch import nn

class logisticRegression(nn.Module): 
    def __init__(self, input_features,num_classes):
        super().__init__()
        # TODO: Please Complete this function
        
        # Initialize a linear layer where the number of input features and output features
        # are defined by the input_features and num_classes parameters, respectively.
        # This layer will compute a linear transformation (y = xA^T + b) on the input data.
        # TODO: Code here

        # Initialize the sequential model
        self.layer_stack = nn.Sequential(
            # Pass through linear layer
            nn.Linear(in_features = input_features, out_features = num_classes),
            # Use softmax to get probability output
            nn.Softmax(dim=1)
        )


        # The softmax function is applied across the columns (dim=1) of the output logits
        # from the linear layer. 
        #TODO: Code 
        


    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        # TODO: Please Complete this function
        # Return output of model
        return self.layer_stack(x) #<--------------- modify this code to return the correct output
    

class multilayerPerceptron(nn.Module):
    def __init__(self,input_features,num_classes,hidden_neurons):
        super().__init__() 
        # TODO: Please Complete this function

        # Constructs a sequential container (nn.Sequential) that will process inputs
        # through multiple layers in sequence. This stack includes a linear layer
        # that transforms inputs to a hidden layer size (hidden_neurons), a ReLU activation
        # function that introduces non-linearity (making the model capable of learning
        # complex patterns), followed by another linear layer that maps the hidden layer
        # representations to the output classes, and another ReLU.

        # TODO: Code here
        # Initialize Sequential model
        self.layer_stack = nn.Sequential(
            # Pass though linear layer
            nn.Linear(in_features=input_features, out_features=hidden_neurons),
            # Pass output through activation function
            nn.ReLU(),
            # Pass through normalization function
            nn.LayerNorm(hidden_neurons),
            # Repeat
            nn.Linear(in_features=hidden_neurons, out_features = hidden_neurons),
            nn.ReLU(),
            nn.LayerNorm(hidden_neurons),
            # Repeat
            nn.Linear(in_features=hidden_neurons, out_features = hidden_neurons),
            nn.ReLU(),
            nn.LayerNorm(hidden_neurons),
            # Repeat
            nn.Linear(in_features=hidden_neurons, out_features = hidden_neurons),
            nn.ReLU(),
            nn.LayerNorm(hidden_neurons),
            # Repeat
            nn.Linear(in_features=hidden_neurons, out_features = hidden_neurons),
            nn.ReLU(),
            nn.LayerNorm(hidden_neurons),
            # Pass through last hidden layer
            nn.Linear(in_features=hidden_neurons, out_features=num_classes),
            # Output from last ReLU
            nn.ReLU()
        )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        # TODO: Please Complete this function
        x = torch.flatten(x, start_dim=1)
        # Return output of model
        return self.layer_stack(x) #<--------------- modify this code to return the correct output

