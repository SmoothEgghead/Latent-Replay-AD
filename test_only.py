import torch
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
from DataPreparation import DataPreparation
from Model import Autoencoder
from ConfigParser import ConfigParser
import time

category = "screw"

config = ConfigParser(os.path.join(os.getcwd(), 'config.yaml'))
meta_config = config.get_config()["meta"]
model_config = config.get_config()["model"]
training_config = config.get_config()["training"]

data_preparation = DataPreparation(meta_config["data_save_path"], category)
_, test_data = data_preparation.get_data()

model = Autoencoder(input_shape=3, hidden_units=model_config["Autoencoder"]["hidden_neurons"])
model.load_state_dict(torch.load("autoencoder.pth"))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

testing_loader = DataLoader(test_data, batch_size=training_config['batch_size'], shuffle=False)


def test():
    model.eval()
    test_loss = []
    labels = []
    with torch.inference_mode():
        for image, label in testing_loader:
            image = image.to(device)
            label = label.to(device)
            test_pred = model(image)

            error_image = ((test_pred - image)**2).mean(dim=1)

            error_area = F.avg_pool2d(error_image.unsqueeze(1), kernel_size=8, stride=1).squeeze(1)

            scores = error_area.amax(dim=[1,2])

            test_loss.extend(scores.cpu().tolist())
            labels.extend(label.cpu().tolist())
            
    return test_loss, labels


t0 = time.perf_counter()
scores, labels = test()
t1 = time.perf_counter()
print("Testing took ", t1-t0, " seconds")

good = test_data.class_to_idx['good']
binary_labels = [0 if label == good else 1 for label in labels]

good_scores = []
anom_scores = []
for i in range (len(scores)):
    if binary_labels[i] == 0:
        good_scores.append(scores[i])
    else:
        anom_scores.append(scores[i])

anoms = 0
total = 0
for i in anom_scores:
    for j in good_scores:
        if i > j:
            anoms += 1
        elif i == j:
            anoms += 0.5
        total += 1
auc = anoms / total

print(category, "AUC: ", auc)