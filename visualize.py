# visualize.py
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Model import FashionMNISTModelV2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = FashionMNISTModelV2(input_shape=1, hidden_units=30, output_shape=1)
model.load_state_dict(torch.load("autoencoder.pth", map_location=device))
model.eval()

testing_data = datasets.FashionMNIST(root='data', train=False, transform=transforms.ToTensor())
testing_loader = DataLoader(testing_data, batch_size=32, shuffle=True)

with torch.inference_mode():
    image, label = next(iter(testing_loader))
    image = image.to(device)
    reconstruction = model(image)

fig, axes = plt.subplots(3, 10, figsize=(20, 4))
for i in range(10):
    axes[0, i].imshow(image[i].cpu().squeeze(), cmap='gray')
    axes[0, i].axis('off')
    axes[1, i].imshow(reconstruction[i].cpu().squeeze(), cmap='gray')
    axes[1, i].axis('off')
   
    score = torch.nn.functional.mse_loss(reconstruction[i], image[i]).item()
    axes[2, i].text(0.5, 0.5, f"{score:.4f}", ha='center', va='center', fontsize=10)
    axes[2, i].axis('off')
axes[0, 0].set_title('Original')
axes[1, 0].set_title('Reconstructed')
plt.tight_layout()
plt.show()