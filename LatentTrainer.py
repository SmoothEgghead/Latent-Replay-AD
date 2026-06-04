import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

def freeze_decoder_snapshot(decoder):
    snapshot = copy.deepcopy(decoder)
    for i in snapshot.parameters():
        i.requires_grad = False
    
    snapshot.eval()
    return snapshot

class LatentTrainer:
    def __init__(self,model, training_data, testing_data, optimizer,
                  criterion, epochs, learning_rate,batch_size, snapshot, buffer):
        self.model = model
        self.training_loader = DataLoader(training_data,batch_size=batch_size,shuffle=True)
        self.testing_loader = DataLoader(testing_data,batch_size=batch_size,shuffle=False)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.criterion = criterion
        self.snapshot = snapshot
        self.buffer = buffer
        self.batch_size = batch_size
        self.model.to(self.device)
        self.snapshot.to(self.device)
        self.set_training_config()

    def set_training_config(self):
        self.optimizer = optim.Adam(self.model.decoder.parameters(), 
                                    lr=self.learning_rate, weight_decay=0)

        self.criterion = nn.MSELoss()

    def get_training_config(self):
        return {
            "model": self.model,
            "training_loader": self.training_loader,
            "testing_loader": self.testing_loader,
            "optimizer": self.optimizer,
            "criterion": self.criterion,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "snapshot": self.snapshot,
            "buffer": self.buffer
        }

    
    def train(self):
        self.model.train()
        self.snapshot.eval()
        
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

            new_loss_total = 0
            latent_loss_total = 0
            loss_total = 0

            for batch, (image, label) in enumerate(self.training_loader):

                image = image.to(self.device)
                label = label.to(self.device)

                self.optimizer.zero_grad()

                new_encoded = self.model.encoder(image)
                new_decoded = self.model.decoder(new_encoded)
                new_loss = self.criterion(image, new_decoded)

                latents = self.buffer.get_latents(self.batch_size*2).to(self.device)
                decoded_latents_old = self.snapshot(latents)
                decoded_latents_new = self.model.decoder(latents)
                latent_loss = self.criterion(decoded_latents_old, decoded_latents_new)

                loss = new_loss + 3*latent_loss
                loss.backward()
                self.optimizer.step()

                new_loss_total += new_loss.item()
                latent_loss_total += latent_loss.item()
                loss_total += loss.item()
                
            print(f"Epoch: {epoch} | New Train Loss: {new_loss_total / (batch + 1):.3f} Latent Train Loss: {latent_loss_total / (batch + 1):.3f}")
        torch.save(self.model.state_dict(), "autoencoder.pth")
        return loss_total
    
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