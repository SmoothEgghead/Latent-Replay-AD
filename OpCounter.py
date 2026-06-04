import torch
from torch import nn
from torchvision.models import resnet50
from Model import Autoencoder
from other_models import logisticRegression
from other_models import multilayerPerceptron
from thop import profile
from thop import clever_format
model = []
input = []
model.append(Autoencoder(input_shape=3, hidden_units=16))
#model = resnet50()
model.append(nn.Conv2d(1, 30, 3, padding=1))
model.append(nn.BatchNorm2d(30))
model.append(nn.Conv2d(30, 30, 3, padding=1))
model.append(nn.Dropout())
model.append(nn.Linear(30*7*7, 512))
model.append(nn.BatchNorm1d(512))
model.append(nn.Linear(512, 32))
model.append(nn.BatchNorm1d(32))
model.append(nn.Linear(32,10))
input.append(torch.randn(16, 3, 256, 256))
input.append(torch.randn(32, 30, 28, 28))
input.append(torch.randn(32, 30, 14, 14))
input.append(torch.randn(32, 30, 7, 7))
input.append(torch.randn(32, 1470))
input.append(torch.randn(32,512))
input.append(torch.randn(32,32))

pairs = [
    (0,0),
    (1,0),
    (2,1),
    (3,2),
    (2,2),
    (3,2),
    (5,4),
    (6,5),
    (7,5),
    (8,6),
    (9,6)
]



model = Autoencoder(input_shape=3, hidden_units=16)
macs, params = profile(model, inputs=(input[0], ))

print(macs, params)

