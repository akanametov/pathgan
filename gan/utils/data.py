import os
import math
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

class MapPointRegionDataset(Dataset):
    '''
    Torch Dataset for Map & Point & Region

    Args:
        map_dir (str): Path to the maps
        point_dir (str): Path to the points
        roi_dir (str): Path to the ROI's
        csv_file (pd.DataFrame): Dataframe with map/task/roi pairs
        transform (torchvision.transforms): Transforms for map/task/roi pairs
    '''
    def __init__(self,
                 map_dir,
                 point_dir,
                 roi_dir,
                 csv_file,
                 transform=None):
        self.map_dir = map_dir
        self.point_dir = point_dir
        self.roi_dir = roi_dir
        self.csv_file = csv_file
        self.transform = transform
        
    def __len__(self,):
        return len(self.csv_file)
    
    def __getitem__(self, idx):
        row = self.csv_file.iloc[idx]
        map_name = row["map"].split('.')[0]
        map_path = f'{self.map_dir}/{row["map"]}'
        point_path = f'{self.point_dir}/{map_name}/{row["task"]}'
        roi_path = f'{self.roi_dir}/{map_name}/{row["roi"]}'
        
        map_img = Image.open(map_path).convert('RGB')
        point_img = Image.open(point_path).convert('RGB')
        roi_img = Image.open(roi_path).convert('RGB')
        
        map_data = self.transform(map_img)
        point_data = self.transform(point_img)
        roi_data = self.transform(roi_img)
        
        return (map_data, point_data, roi_data)
