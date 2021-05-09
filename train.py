import os
import argparse

import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, ToPILImage, Normalize

from gan.utils.data import MapPointRegionDataset as MPRDataset
from gan.generator import Generator
from gan.discriminator import MapDiscriminator, PointDiscriminator
from gan.utils.criterion import AdaptiveGeneratorLoss, DiscriminatorLoss
from gan.utils.trainer import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = 'top', description='Training GAN (from original paper')
    
    parser.add_argument('--batch_size', type=int, default=8,
                        help='"Batch size" with which GAN will be trained (default: 8)')
    
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of "epochs" GAN will be trained (default: 3)')
    
    parser.add_argument('--g_lr', type=float, default=0.0001,
                        help='"Learning rate" of Generator (default: 0.0001)')
    
    parser.add_argument('--md_lr', type=float, default=0.00005,
                        help='"Learning rate" of Map Discriminator (default: 0.00005)')
    
    parser.add_argument('--pd_lr', type=float, default=0.00005,
                        help='"Learning rate" of Point Discriminator (default: 0.00005)')
    
    parser.add_argument('--load_dir', default=None,
                        help='Load directory to continue training (default: "None")')
    
    parser.add_argument('--save_dir', default=None,
                        help='Save directory (default: "gan/checkpoint/generator.pth")')
    
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform = Compose([ToTensor(),
                        Normalize(mean=(0.5, 0.5, 0.5),
                                  std=(0.5, 0.5, 0.5))])

    train_df = pd.read_csv('dataset/train.csv')

    trainset = MPRDataset(map_dir = 'dataset/maps',
                          point_dir = 'dataset/tasks',
                          roi_dir = 'dataset/tasks',
                          csv_file = train_df,
                          transform = transform)

    traingen = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    generator = Generator()
    map_discriminator = MapDiscriminator()
    point_discriminator = PointDiscriminator()
    
    if args.load_dir:
        print('=========== Loading weights for Generator ===========')
        generator.load_state_dict(torch.load(args.load_dir))

    g_criterion = AdaptiveGeneratorLoss()
    md_criterion = DiscriminatorLoss()
    pd_criterion = DiscriminatorLoss()

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
    md_optimizer = torch.optim.Adam(generator.parameters(), lr=args.md_lr, betas=(0.5, 0.999))
    pd_optimizer = torch.optim.Adam(generator.parameters(), lr=args.pd_lr, betas=(0.5, 0.999))
    
    trainer = Trainer(generator = generator,
                      map_discriminator = map_discriminator,
                      point_discriminator = point_discriminator,
                      g_criterion = g_criterion,
                      md_criterion = md_criterion,
                      pd_criterion = pd_criterion,
                      g_optimizer = g_optimizer,
                      md_optimizer = md_optimizer,
                      pd_optimizer = pd_optimizer,
                      device = device)
    print('============== Training Started ==============')
    trainer.fit(traingen, epochs=args.epochs, device=device)
    print('============== Training Finished! ==============')
    if args.save_dir:
        print('=========== Saving weights for Generator ===========')
        torch.save(generator.cpu().state_dict(), args.save_dir)
