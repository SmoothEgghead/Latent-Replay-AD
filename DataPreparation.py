import torch
import numpy as np
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

class DataPreparation:

    def __init__(self, data_save_path, category):

        
        self.data_save_path = data_save_path
        self.category = category
        self.width = 256
        self.height = 256
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def get_data(self):
        training_data = datasets.ImageFolder(
            self.data_save_path + self.category + "/train",
            transform = self.transform
        )
        testing_data = datasets.ImageFolder(
            self.data_save_path + "/" + self.category + "/test",
            transform = self.transform
        )
        return training_data, testing_data