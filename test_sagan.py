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
from pathgan.metrics import intersection_over_union, jaccard_coefficient


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = 'top', description='Testing SAGAN (from original paper)')
    parser.add_argument('--checkpoint_path', default='checkpoints/sagan/generator.pt', help='Path to trained Generator')
    parser.add_argument('--dataset_path', default='data/generated_dataset/dataset', help='Path to dataset')
    parser.add_argument('--save_dir', default='results/sagan', help='Save directory (default: "results/sagan")')
    parser.add_argument('--device', type=str, default='cuda', help='Device (default: "cuda")')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

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
        csv_file=pd.read_csv(os.path.join(args.dataset_path, 'test.csv')),
        transform=transform,
        return_meta=True,
    )
    generator = SAGenerator()
    print(f"Loading weights from: {args.checkpoint_path}")
    generator.load_state_dict(torch.load(args.checkpoint_path, map_location="cpu"))
    generator = generator.to(device)
    generator = generator.eval()
    print("Start evaluation")
    os.makedirs(args.save_dir, exist_ok=True)
    true_roi_paths = []
    pred_roi_paths = []
    iou_values = []
    dice_values = []
    for i in tqdm(range(len(dataset))):
        maps, points, rois, meta = dataset[i]
        maps = maps.unsqueeze(0).to(device)
        points = points.unsqueeze(0).to(device)
        b, _, h, w = maps.size() 
        noise = torch.rand(b, 1, h, w)
        noise = noise.to(device)
        with torch.no_grad():
            pred_rois = generator(maps, points, noise).detach().cpu()[0]
        pred_rois = pred_rois.permute(1,2,0).numpy()
        pred_rois = (pred_rois > 0).astype(np.uint8) * 255

        pred_roi_img = Image.fromarray(pred_rois)
        pred_roi_path = os.path.join(args.save_dir, f"roi_{i}.png")
        pred_roi_img.save(pred_roi_path)

        iou = intersection_over_union(pred_rois, rois.permute(1,2,0).numpy())
        dice = jaccard_coefficient(pred_rois, rois.permute(1,2,0).numpy())
        true_roi_paths.append(meta["roi_path"])
        pred_roi_paths.append(pred_roi_path)
        iou_values.append(iou)
        dice_values.append(dice)

    csv_file = pd.DataFrame({
        "true_roi": true_roi_paths,
        "pred_roi": pred_roi_paths,
        "iou": iou_values,
        "dice": dice_values,
    })
    csv_file.to_csv(os.path.join(args.save_dir, "results.csv"), index=False)
    print(f"Saving .csv file to: {args.save_dir}")
