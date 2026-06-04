import torch
from torch import nn
class Encoder(nn.Module):
    def __init__(self, input_shape = 3, hidden_units = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
            out_channels=hidden_units,
            kernel_size=4,
            stride=2,
            padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
        
            nn.Conv2d(hidden_units, hidden_units, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),

            nn.Conv2d(hidden_units, hidden_units, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),

            nn.Conv2d(hidden_units, hidden_units, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor):
        return self.encoder(x) #<------------ Replace None with the output of the model
    
class Decoder(nn.Module):
    def __init__(self, input_shape = 3, hidden_units = 16):
        super().__init__()
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(hidden_units, hidden_units, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),

            nn.ConvTranspose2d(hidden_units, hidden_units, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),

            nn.ConvTranspose2d(hidden_units, hidden_units, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),

            nn.ConvTranspose2d(hidden_units, input_shape, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        return self.decoder(x)
    
class Autoencoder(nn.Module):
    def __init__(self, input_shape = 3, hidden_units = 16):
        super().__init__()
        self.encoder = Encoder(input_shape, hidden_units)
        self.decoder = Decoder(input_shape, hidden_units)

    def forward(self, x: torch.Tensor):
        return self.decoder(self.encoder(x))
    
