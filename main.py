import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import logging
from DataPreparation import DataPreparation
from Model import Autoencoder
from ConfigParser import ConfigParser
from Trainer import Trainer
from Logger import Logger
from Plot import Plot
from ReplayBuffer import ReplayBuffer
import time

torch.manual_seed(42)

if os.path.exists("mylog.log"):
    log_file_size = os.path.getsize("mylog.log")
    if log_file_size > 2 * 1024 * 1024:
        os.remove("mylog.log")
        print("The log file was greater than 2MB and has been deleted.")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename="mylog.log",
    filemode='a'
)

stdout_logger = logging.getLogger('STDOUT')
sys.stdout = Logger(stdout_logger, logging.DEBUG)
config = ConfigParser(os.path.join(os.getcwd(), 'config.yaml'))
meta_config = config.get_config()["meta"]
model_config = config.get_config()["model"]

"""
Data Preparation
"""
category = "hazelnut"
data_preparation = DataPreparation(meta_config["data_save_path"], category)
train_data, test_data = data_preparation.get_data()
image_width = data_preparation.width
image_height = data_preparation.height
image_depth = 3
"""
Hyperparameters preparation
"""
config = ConfigParser(os.path.join(os.getcwd(), 'config.yaml'))
training_config = config.get_config()["training"]
model_config = config.get_config()["model"]
if training_config['model'] == "Autoencoder":
    model = Autoencoder(input_shape=image_depth,hidden_units=model_config["Autoencoder"]["hidden_neurons"])
else:
    raise ValueError("Model not supported")
training_config['model'] = model
training_config['training_data'] = train_data
training_config['testing_data'] = test_data


"""
Training and Testing
"""
trainer = Trainer(**training_config)

batches_per_epoch = len(trainer.training_loader)
trainer.epochs = max(1, 125 // batches_per_epoch)
print("# Epochs: ", trainer.epochs)

t0 = time.perf_counter()
trainer.train()
t1 = time.perf_counter()
print("Training took ", t1-t0, " seconds")

t0 = time.perf_counter()
scores, labels = trainer.test()
t1 = time.perf_counter()
print("Testing took ", t1-t0, " seconds")

scores_tensor = torch.tensor(scores)
labels_tensor = torch.tensor(labels)
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

buffer = ReplayBuffer()
model.eval()
idx = torch.randperm(len(train_data))[:200]
images = torch.stack([train_data[i][0] for i in idx]).to(device)
with torch.inference_mode():
    latents = model.encoder(images)
    
buffer.add_latent(latents, category)
buffer.save_latents("buffer.pt")

"""
Plotting

"""
Plot().plot_prediction(model, test_data, os.path.join(os.getcwd(), "predictions.png"))

