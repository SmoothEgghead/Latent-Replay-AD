import torch

class ReplayBuffer:
    def __init__(self):
        self.latent_vectors = {}

    def add_latent(self, latents, category):
        self.latent_vectors[category] = latents.detach().cpu()

    def get_latents(self, num_latents):
        if len(self.latent_vectors) == 0:
            return None
        
        all_latents = torch.cat(list(self.latent_vectors.values()), dim=0)
        total_latents = all_latents.shape[0]
        num_latents = min(total_latents, num_latents)

        idx = torch.randperm(total_latents)[:num_latents]
        return all_latents[idx]
    
    def __len__(self):
        return sum(t.shape[0] for t in self.latent_vectors.values())
    
    def get_categories(self):
        return list(self.latent_vectors.keys())
    
    def save_latents(self, path):
        torch.save(self.latent_vectors, path)

    def load_latents(self, path):
        self.latent_vectors = torch.load(path)