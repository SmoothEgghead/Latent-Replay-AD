import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


class Plot:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def plot_prediction(self, model, data, figname, n=8):
        model.eval()
        model.to(self.device)
        loader = DataLoader(data, batch_size=n, shuffle=True)
        images, labels = next(iter(loader))
        images = images.to(self.device)

        with torch.no_grad():
            reconstructions = model(images)
            errors = ((images - reconstructions) ** 2).mean(dim=[1, 2, 3])

        # tensors are [B, C, H, W]; matplotlib wants [H, W, C]
        imgs_np = images.cpu().permute(0, 2, 3, 1).numpy()
        recon_np = reconstructions.cpu().permute(0, 2, 3, 1).numpy()

        fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.5))
        for i in range(n):
            axes[0, i].imshow(imgs_np[i].clip(0, 1))
            axes[0, i].set_title(f"input (lbl={labels[i].item()})", fontsize=8)
            axes[0, i].axis('off')

            axes[1, i].imshow(recon_np[i].clip(0, 1))
            axes[1, i].set_title(f"recon mse={errors[i].item():.4f}", fontsize=8)
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.savefig(figname)
        plt.close()