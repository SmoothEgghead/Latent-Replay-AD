import unittest
import os
import torch
from DataPreparation import DataPreparation
from Model import Autoencoder
from Trainer import Trainer
from ConfigParser import ConfigParser

class Hoemwork4Test(unittest.TestCase):


    def test_model(self):
        config = ConfigParser(os.path.join(os.getcwd(), 'config.yaml'))
        meta_config = config.get_config()["meta"]
        cnn_model = Autoencoder(input_shape=3,hidden_units=16)
        data_preparation = DataPreparation(meta_config["data_save_path"])
        train_data, test_data = data_preparation.get_data()
        images, labels = next(iter(test_data))
        images = images.unsqueeze_(0)
        test_inference = cnn_model(images)
        self.assertIsNotNone(test_inference,"The model prediction output should not be None")
        
 
    def test_training(self):
        config = ConfigParser(os.path.join(os.getcwd(), 'config.yaml'))
        training_config = config.get_config()["training"]
        meta_config = config.get_config()["meta"]
        model_config = config.get_config()["model"]
        cnn_model = Autoencoder(input_shape=3,hidden_units=30,output_shape=10)        
        data_preparation = DataPreparation(meta_config["data_save_path"])
        train_data, test_data = data_preparation.get_data()
        training_config['model'] = cnn_model
        training_config['training_data'] = train_data
        training_config['testing_data'] = test_data
        trainer = Trainer(**training_config)
        training_loss = trainer.train()
        testing_loss = trainer.test()
        self.assertIsNotNone(training_loss)
        self.assertIsNotNone(training_loss)
        self.assertLess(testing_loss,0.25)


if __name__ == '__main__':
    unittest.main()