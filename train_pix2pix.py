import os
import argparse

import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, ToPILImage, Normalize

from pathgan.data import MPRDataset
from pathgan.models import Generator, Discriminator
from pathgan.losses import GeneratorLoss, DiscriminatorLoss
from pathgan.train import Pix2PixTrainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = 'top', description='Training Pix2Pix GAN (our GAN)')
    parser.add_argument('--dataset_path', default='data/generated_dataset/dataset', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='"Batch size" with which GAN will be trained (default: 8)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of "epochs" GAN will be trained (default: 3)')
    parser.add_argument('--g_lr', type=float, default=0.001, help='"Learning rate" of Generator (default: 0.001)')
    parser.add_argument('--d_lr', type=float, default=0.0007, help='"Learning rate" of Discriminator (default: 0.0007)')
    parser.add_argument("--save_dir", default="checkpoints/pix2pix", help='Save directory (default: "checkpoints/pix2pix")')
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (default: 'cuda:0')")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    transform = Compose([
        ToTensor(),
        Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])
    dataset = MPRDataset(
        map_dir=os.path.join(args.dataset_path, 'maps'),
        point_dir=os.path.join(args.dataset_path, 'tasks'),
        roi_dir=os.path.join(args.dataset_path, 'tasks'),
        csv_file=pd.read_csv(os.path.join(args.dataset_path, 'train.csv')),
        transform=transform,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    generator = Generator()
    discriminator = Discriminator()

    g_criterion = GeneratorLoss()
    d_criterion = DiscriminatorLoss()
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
    
    trainer = Pix2PixTrainer(
        generator = generator,
        discriminator=discriminator,
        g_criterion=g_criterion,
        d_criterion=d_criterion,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        device=device,
    )
    print("Start training")
    trainer.fit(dataloader, epochs=args.epochs, device=device)
    if args.save_dir:
        print(f"Saving weights for Pix2pix to: {args.save_dir}")
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(generator.cpu().state_dict(), os.path.join(args.save_dir, "generator.pt"))
        torch.save(discriminator.cpu().state_dict(), os.path.join(args.save_dir, "discriminator.pt"))