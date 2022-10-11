import os
import argparse

import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T

from pathgan.data import MPRDataset
from pathgan.models import SAGenerator, MapDiscriminator, PointDiscriminator
from pathgan.losses import AdaptiveSAGeneratorLoss, DiscriminatorLoss
from pathgan.train import SAGANTrainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "top", description="Training GAN (from original paper)")
    parser.add_argument('--dataset_path', default='data/generated_dataset/dataset', help='Path to dataset')
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of `epochs` GAN will be trained (default: 3)")
    parser.add_argument("--g_lr", type=float, default=0.0001, help="Learning rate of Generator (default: 0.0001)")
    parser.add_argument("--md_lr", type=float, default=0.00005, help="Learning rate of Map Discriminator (default: 0.00005)")
    parser.add_argument("--pd_lr", type=float, default=0.00005, help="Learning rate of Point Discriminator (default: 0.00005)")
    parser.add_argument("--save_dir", default="checkpoints/sagan", help='Save directory (default: "checkpoints/sagan")')
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (default: 'cuda:0')")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # Dataset
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(
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
    # Models
    generator = SAGenerator()
    map_discriminator = MapDiscriminator()
    point_discriminator = PointDiscriminator()
    # Losses
    g_criterion = AdaptiveSAGeneratorLoss()
    md_criterion = DiscriminatorLoss()
    pd_criterion = DiscriminatorLoss()
    # Optimizers
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
    md_optimizer = torch.optim.Adam(generator.parameters(), lr=args.md_lr, betas=(0.5, 0.999))
    pd_optimizer = torch.optim.Adam(generator.parameters(), lr=args.pd_lr, betas=(0.5, 0.999))
    # Pipeline
    trainer = SAGANTrainer(
        generator=generator,
        map_discriminator=map_discriminator,
        point_discriminator=point_discriminator,
        g_criterion=g_criterion,
        md_criterion=md_criterion,
        pd_criterion=pd_criterion,
        g_optimizer=g_optimizer,
        md_optimizer=md_optimizer,
        pd_optimizer=pd_optimizer,
        device=device,
    )
    print("Start training")
    trainer.fit(dataloader, epochs=args.epochs, device=device)
    if args.save_dir:
        print(f"Saving weights for SAGAN to: {args.save_dir}")
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(generator.cpu().state_dict(), os.path.join(args.save_dir, "generator.pt"))
        torch.save(map_discriminator.cpu().state_dict(), os.path.join(args.save_dir, "map_discriminator.pt"))
        torch.save(point_discriminator.cpu().state_dict(), os.path.join(args.save_dir, "point_discriminator.pt"))
