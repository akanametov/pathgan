import os
import math
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from data.utils import rgb2binary
from pathlib import Path

class ROIGenerator:
    def __init__(self, model):
        self.model = model
        self.colors = {'roi': np.array([0, 1, 0]), 'start': np.array([0, 0, 1]), 'goal': np.array([1, 0, 0])}
        
    def set_parameters(self, m_name=None, m_path='data/dataset/maps/', t_path='data/dataset/tasks/'):
        self.m_name = m_name # with extantion
        self.fname = Path(m_name).stem # without extantion
        Map=Image.open(m_path + self.m_name).convert('RGB')
        Map = np.array(Map)
        self.Map = rgb2binary(Map)
        self.MapTasks = pd.read_csv(t_path + self.fname + '.csv')
        self.m_path = m_path
        self.t_path = t_path
        
    def generate(self, n_tasks=100, n_runs=50):
        grids, rois = [], []

        for t in range(n_tasks):
            start = (self.MapTasks.istart[t], self.MapTasks.jstart[t])
            goal = (self.MapTasks.igoal[t], self.MapTasks.jgoal[t])

            grid = self.Map.copy()
            roi = np.ones(grid.shape)

            try:
                for _ in range(n_runs):
                    path = self.model.search(self.Map[..., 1], start, goal, func='cdist')
                    for x in path:
                        grid[x[0], x[1], :] = self.colors['roi']
                        roi[x[0], x[1], :] = self.colors['roi']
            except TypeError:
                continue

            grid[start[0], start[1], :] = self.colors['start']
            grid[goal[0], goal[1], :] = self.colors['goal']
            grids.append(grid)
            rois.append(roi)
            plt.imsave(self.t_path + self.fname + f'/task_{t}_rrt.png', grid)
            plt.imsave(self.t_path + self.fname + f'/task_{t}_roi.png', roi)
        return grids, rois