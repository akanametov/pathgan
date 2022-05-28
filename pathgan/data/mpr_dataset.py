"""MPRDataset."""

from typing import Any, Dict, Callable, Optional

import numpy as np
import os
import math
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


class MPRDataset(Dataset):
    """MPRDataset.

    Torch Dataset for Map & Point & Region

    Parameters
    ----------
    map_dir: str
        Path to the maps.
    point_dir: str
        Path to the points.
    roi_dir: str
        Path to the ROI's.
    csv_file: pd.DataFrame
        Dataframe with map/task/roi pairs.
    transform: Callable
        Transforms for map/task/roi pairs.
    """
    def __init__(
        self,
        map_dir: str,
        point_dir: str,
        roi_dir: str,
        csv_file: pd.DataFrame,
        transform: Optional[Callable] = None,
        test: bool = False,
    ):
        self.map_dir = map_dir
        self.point_dir = point_dir
        self.roi_dir = roi_dir
        self.csv_file = csv_file
        self.transform = transform
        self.test = test

    def __len__(self) -> int:
        return len(self.csv_file)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.csv_file.iloc[index]
        map_name = row["map"].split(".")[0]
        map_path = f"{self.map_dir}/{row['map']}"
        point_path = f"{self.point_dir}/{map_name}/{row['task']}"
        if not self.test:
            roi_path = f"{self.roi_dir}/{map_name}/{row['roi']}"

        map_img = np.array(Image.open(map_path).convert('RGB'))
        point_img = np.array(Image.open(point_path).convert('RGB'))
        if not self.test:
            roi_img = np.array(Image.open(roi_path).convert('RGB'))

        if self.transform is not None:
            map_img = self.transform(map_img)
            point_img = self.transform(point_img)
            if not self.test:
                roi_img = self.transform(roi_img)
        if not self.test:
            return map_img, point_img, roi_img
        return map_img, point_img
