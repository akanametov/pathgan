import torch
from torch import nn
from progress.bar import IncrementalBar
from pathgan.losses.criterion import PixelwiseLossMSE 


class SAGANTrainer():
    """SAGAN Trainer.

    Parameters
    ----------
    generator: nn.Module
        Generator model of GAN.
    map_discriminator: nn.Module
        Map Discriminator model of GAN.
    point_discriminator: nn.Module
        Point Discriminator model of GAN.
    g_criterion: nn.Module
        Criterion for Generator.
    md_criterion: nn.Module
        Criterion for Map Discriminator.
    pd_criterion: nn.Module
        Criterion for Point Discriminator.
    g_optimizer: torch.optim.Optimizer
        Optimizer for Generator.
    md_optimizer: torch.optim.Optimizer
        Optimizer for Map Discriminator.
    pd_optimizer: torch.optim.Optimizer
        Optimizer for Point Discriminator.
    device: torch.device
        Device for models.
    """
    def __init__(
        self,
        generator: nn.Module,
        map_discriminator: nn.Module,
        point_discriminator: nn.Module,
        g_criterion: nn.Module,
        md_criterion: nn.Module,
        pd_criterion: nn.Module,
        g_optimizer: torch.optim.Optimizer,
        md_optimizer: torch.optim.Optimizer,
        pd_optimizer: torch.optim.Optimizer,
        device: str = "cuda:0",
    ):
        self.generator = generator.to(device)
        self.map_discriminator = map_discriminator.to(device)
        self.point_discriminator = point_discriminator.to(device)
        self.g_criterion = g_criterion.to(device)
        self.md_criterion = md_criterion.to(device)
        self.pd_criterion = pd_criterion.to(device)
        self.g_optimizer = g_optimizer
        self.md_optimizer = md_optimizer
        self.pd_optimizer = pd_optimizer
        self.pix_loss = PixelwiseLossMSE()
        
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
        md_losses=[]
        pd_losses=[]
        i = 0
        for epoch in range(epochs):
            ge_loss=0.
            mde_loss=0.
            pde_loss=0.
            bar = IncrementalBar(f"Epoch {epoch+1}/{epochs}:", max=len(dataloader))
            for real_map, real_point, real_roi in dataloader:
                b, _, h, w = real_map.size() 
                noise = torch.rand(b, 1, h, w)
                real_map = real_map.to(device)
                real_point = real_point.to(device)
                real_roi = real_roi.to(device)
                noise = noise.to(device)

                # Map Discriminator`s loss
                fake_roi = self.generator(real_map, real_point, noise).detach()
                fake_roimap_pred = self.map_discriminator(real_map, fake_roi)
                real_roimap_pred = self.map_discriminator(real_map, real_roi)
                map_loss = self.md_criterion(fake_roimap_pred, real_roimap_pred)

                # Point Discriminator`s loss
                fake_roipoint_pred = self.point_discriminator(real_point, fake_roi)
                real_roipoint_pred = self.point_discriminator(real_point, real_roi)
                point_loss = self.pd_criterion(fake_roipoint_pred, real_roipoint_pred)

                # Map Discriminator`s params update
                self.md_optimizer.zero_grad()
                map_loss.backward()
                self.md_optimizer.step()

                # Point Discriminator`s params update
                self.pd_optimizer.zero_grad()
                point_loss.backward()
                self.pd_optimizer.step()

                # Generator`s loss
                fake_roi = self.generator(real_map, real_point, noise)
                fake_roimap_pred = self.map_discriminator(real_map, fake_roi)
                fake_roipoint_pred = self.point_discriminator(real_point, fake_roi)
                gan_loss = self.g_criterion(fake_roimap_pred, fake_roipoint_pred, map_loss.item(), point_loss.item())
                gen_loss = gan_loss + self.pix_loss(fake_roi, real_roi)

                # Generator`s params update
                self.g_optimizer.zero_grad()
                gen_loss.backward()
                self.g_optimizer.step()

                ge_loss += gen_loss.item()
                mde_loss += map_loss.item()
                pde_loss += point_loss.item()
                bar.next()
            bar.finish()    
            g_losses.append(ge_loss/len(dataloader))
            md_losses.append(mde_loss/len(dataloader))
            pd_losses.append(pde_loss/len(dataloader))
        self.data={"g_loss": g_losses, "md_loss": md_losses, "pd_loss": pd_losses}
