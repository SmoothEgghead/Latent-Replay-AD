import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self,model, training_data, testing_data, optimizer, criterion, epochs, learning_rate,batch_size):
        self.model = model
        self.training_loader = DataLoader(training_data,batch_size=batch_size,shuffle=True)
        self.testing_loader = DataLoader(testing_data,batch_size=batch_size,shuffle=False)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.criterion = criterion
        self.model.to(self.device)
        self.set_training_config()

    def set_training_config(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0)

        self.criterion = nn.MSELoss()

    def get_training_config(self):
        return {
            "model": self.model,
            "training_loader": self.training_loader,
            "testing_loader": self.testing_loader,
            "optimizer": self.optimizer,
            "criterion": self.criterion,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate
        }

    
    def train(self):
        self.model.train()
        
        for epoch in range(self.epochs):
            train_loss = 0
            if epoch < 10:
                lr = 0.005
            elif epoch < 20:
                lr = 0.001
            else:
                lr = 0.0001
            for i in self.optimizer.param_groups:
                    i['lr'] = lr
            for batch, (image, label) in enumerate(self.training_loader):

                image = image.to(self.device)
                label = label.to(self.device)

                self.optimizer.zero_grad()
                y_pred = self.model(image)
                loss = self.criterion(y_pred, image)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                
            print(f"Epoch: {epoch} | Train Loss: {train_loss / (batch + 1):.3f}")
        torch.save(self.model.state_dict(), "autoencoder.pth")
        return train_loss
    
    def test(self):
        self.model.eval()
        test_loss = []
        labels = []
        with torch.inference_mode():
            for image, label in self.testing_loader:
                image = image.to(self.device)
                label = label.to(self.device)
                test_pred = self.model(image)

                error_image = ((test_pred - image)**2).mean(dim=1)

                error_area = F.avg_pool2d(error_image.unsqueeze(1), kernel_size=8, stride=1).squeeze(1)

                scores = error_area.amax(dim=[1,2])

                test_loss.extend(scores.cpu().tolist())
                labels.extend(label.cpu().tolist())
                
        return test_loss, labels