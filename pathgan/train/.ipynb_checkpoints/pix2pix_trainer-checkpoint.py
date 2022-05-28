import torch
from torch import nn
from progress.bar import IncrementalBar


class Pix2PixTrainer():
    """GAN Trainer.

    Parameters
    ----------
    generator: nn.Module
        Generator model of GAN.
    discriminator: nn.Module
        Discriminator model of GAN.
    g_criterion: nn.Module
        Criterion for Generator.
    d_criterion: nn.Module
        Criterion for Discriminator.
    g_optimizer: torch.optim.Optimizer
        Optimizer for Generator.
    d_optimizer: torch.optim.Optimizer
        Optimizer for Discriminator.
    device: torch.device
        Device for models.
    """
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        g_criterion: nn.Module,
        d_criterion: nn.Module,
        g_optimizer: torch.optim.Optimizer,
        d_optimizer: torch.optim.Optimizer,
        device: str = "cuda:0",
    ):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.g_criterion = g_criterion.to(device)
        self.d_criterion = d_criterion.to(device)
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        
    def fit(self, dataloader, epochs: int = 10, device: str = "cuda:0"):
        """Run trainer.

        Parameters
        ----------
        dataloader: torch.data.Dataloader
            Train dataloader.
        epochs: int, (default=10)
            Number of epochs for train.
        device: torch.device
            Device for data.

        Returns
        -------
        Dict[str, Any]:
            Dictionary with losses.
        """
        g_losses=[]
        d_losses=[]

        for epoch in range(epochs):
            ge_loss=0.
            de_loss=0.
            bar = IncrementalBar(f"Epoch {epoch+1}/{epochs}:", max=len(dataloader))
            for maps, points, real in dataloader:
                maps, points, real = maps.to(device), points.to(device), real.to(device)

                # Generator`s loss
                fake = self.generator(maps, points)
                fake_pred = self.discriminator(torch.cat([maps, points, fake], 1))
                g_loss = self.g_criterion(fake, real, fake_pred)

                # Discriminator`s loss
                fake = self.generator(maps, points).detach()
                fake_pred = self.discriminator(torch.cat([maps, points, fake], 1))
                real_pred = self.discriminator(torch.cat([maps, points, real], 1))
                d_loss = self.d_criterion(fake_pred, real_pred)

                # Generator`s params update
                self.g_optimizer.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Discriminator`s params update
                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                ge_loss += g_loss.item()
                de_loss += d_loss.item()
                bar.next()
            bar.finish() 
            g_losses.append(ge_loss/len(dataloader))
            d_losses.append(de_loss/len(dataloader))
        self.data={"g_loss": g_losses, "d_loss": d_losses}
