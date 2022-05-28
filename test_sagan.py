import os
import argparse

import pandas as pd
import numpy as np
from PIL import Image

from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T

from pathgan.data import MPRDataset
from pathgan.models import SAGenerator


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = 'top', description='Testing SAGAN (from original paper)')
    parser.add_argument('--checkpoint_path', default=None, help='Load directory to continue training (default: "None")')
    parser.add_argument('--batch_size', type=int, default=1, help='"Batch size" with which GAN will be trained (default: 1)')
    parser.add_argument('--save_dir', default='results/sagan', help='Save directory (default: "results/sagan")')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device (default: "cuda:0")')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])
    df = pd.read_csv('dataset/test.csv')
    dataset = MPRDataset(
        map_dir = 'dataset/maps',
        point_dir = 'dataset/tasks',
        roi_dir = 'dataset/tasks',
        csv_file = df,
        transform = transform,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    generator = SAGenerator()
    print('=========== Loading weights for Generator ===========')
    generator.load_state_dict(torch.load(args.checkpoint_path, map_location="cpu"))
    generator = generator.to(device)
    generator = generator.eval()
    print('============== Testing Started ==============')
    os.makedirs(args.save_dir, exist_ok=True)
    for i, (maps, points, rois) in enumerate(tqdm(dataloader)):
        maps = maps.to(device)
        points = points.to(device)
        b, _, h, w = maps.size() 
        noise = torch.rand(b, 1, h, w)
        noise = noise.to(device)
        with torch.no_grad():
            pred_rois = generator(maps, points, noise).detach().cpu()[0]
        pred_rois = pred_rois.permute(1,2,0).numpy()
        pred_rois = (pred_rois > 0).astype(np.uint8) * 255
        roi_img = Image.fromarray(pred_rois)
        roi_path = os.path.join(args.save_dir, f"roi_{i}.png")
        roi_img.save(roi_path)
    print('============== Testing Finished! ==============')
