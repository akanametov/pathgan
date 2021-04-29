import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from tqdm.notebook import tqdm
from IPython.display import clear_output

from .criterion import PixelwiseLossMSE 

######################################
####### Function to show images  #####
######################################

def showImage(img_data, title):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    img_data = img_data.detach().cpu()
    img_data = torch.clip(img_data, 0, 1)
    img_grid = make_grid(img_data[:4], nrow=4)
    plt.axis('off')
    plt.title(title)
    plt.imshow(img_grid.permute(1, 2, 0).squeeze())
    plt.show()
    
    
class Trainer():
    def __init__(self,
                 generator,
                 map_discriminator,
                 point_discriminator,
                 g_criterion,
                 md_criterion,
                 pd_criterion,
                 g_optimizer,
                 md_optimizer,
                 pd_optimizer,
                 device='cuda:0'):
        
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
        
    def fit(self, dataloader, epochs=500, device='cuda:0'):
        g_losses=[]
        md_losses=[]
        pd_losses=[]
        i = 0
        for epoch in range(epochs):
            ge_loss=0.
            mde_loss=0.
            pde_loss=0.
            for k, (real_map, real_point, real_roi) in enumerate(tqdm(dataloader)):
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
                gen_loss = self.g_criterion(fake_roimap_pred, fake_roipoint_pred, map_loss.item(), point_loss.item())\
                         + self.pix_loss(fake_roi, real_roi)
                    
                # Generator`s params update
                self.g_optimizer.zero_grad()
                gen_loss.backward()
                self.g_optimizer.step()
                
                ge_loss += gen_loss.item()
                mde_loss += map_loss.item()
                pde_loss += point_loss.item()
                if i%200 ==0:
                    clear_output(wait=True)
                    print(f'::::::::::  Epoch {epoch+1}  :::: Iteration {i}  :::::::::::')
                    print(f'::::::::::: Generator loss: {gen_loss.item():.3f} :::::::::::')
                    print(f'::::::::: Map Discriminator loss: {map_loss.item():.3f} :::::::::')
                    print(f'::::::::: Point Discriminator loss: {point_loss.item():.3f} :::::::::')
                    showImage(fake_roi.to(real_roi.dtype), 'Generated Output Image')
                    showImage(real_roi, 'Real Output Image')
                i += 1
                
            g_losses.append(ge_loss/len(dataloader))
            md_losses.append(mde_loss/len(dataloader))
            pd_losses.append(pde_loss/len(dataloader))
        self.data={'g_loss': g_losses, 'md_loss': md_losses, 'pd_loss': pd_losses}