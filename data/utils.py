import math
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def rgb2binary(img):
    return (img[..., 1] > 150).astype(int)

def save_maps_and_tasks(maps, tasks, m_path, t_path, istart=0):
    names = list(range(istart, istart+len(maps)+1))
    for i, (m, t) in enumerate(zip(maps, tasks)):
        plt.imsave(m_path+f'map_{names[i]}.jpg', m, cmap='gray')
        pd.DataFrame.from_dict(t).to_csv(t_path+f'map_{names[i]}.csv', index=False)

def make_grid(maps, n_maps=5, margin=5):
    H, W = maps[0].shape
    grid = np.zeros((H+2*margin, n_maps*(W + margin) + margin))+0.3
    for i, m in enumerate(maps[:n_maps]):
        grid[margin:-margin, i*W +(i+1)*margin : (i+1)*W + (i+1)*margin] = m
    return grid

class MapAugmentator():
    def __init__(self, init_map):
        self.init_map = init_map
        
    def set_parameters(self, h_shift, w_shift, step, t_prob):
        self.h_shift = h_shift
        self.w_shift = w_shift
        self.h_range = np.arange(- h_shift, h_shift, step)
        self.w_range = np.arange(- w_shift, w_shift, step)
        self.step=step
        self.t_prob = t_prob
        
    def generate(self, n_maps=10):
        maps = []
        H, W = self.init_map.shape
        nH = H + 2*self.h_shift
        nW = W + 2*self.w_shift
        
        for n in range(n_maps):
            grid = np.ones((nH, nW))
            
            h = np.random.choice(self.h_range, 1).item()
            w = np.random.choice(self.w_range, 1).item()
            
            grid[self.h_shift + self.step*h: H + self.h_shift + self.step*h,
                 self.w_shift + self.step*w: W + self.w_shift + self.step*w] = self.init_map
            g_map = grid[self.h_shift: - self.h_shift, self.w_shift: - self.w_shift].copy()
            if np.random.rand(1) < self.t_prob:
                g_map = g_map.T
            assert g_map.shape == (H, W)
            maps.append(g_map)
        return maps
    
class TaskGenerator():
    def __init__(self, gmap):
        self.gmap=gmap
        
    def set_parameters(self, min_length):
        self.min_length=min_length
        
    def euclid(self, i1, j1, i2, j2):
        return math.sqrt((i1 - i2)**2 + (j1 - j2)**2)
        
    def generate(self, n_tasks):
        tasks =  {'istart': [],
                  'jstart': [],
                  'igoal' : [],
                  'jgoal' : [],
                  'euclid': []}
        H, W = self.gmap.shape
        t=0
        while t < n_tasks:
            istart = np.random.choice(H, 1).item()
            jstart = np.random.choice(W, 1).item()
            igoal = np.random.choice(H, 1).item()
            jgoal = np.random.choice(W, 1).item()
            traversable=(self.gmap[istart, jstart] != 0) and (self.gmap[igoal, jgoal] != 0)
            dist = self.euclid(istart, jstart, igoal, jgoal)
            goodlength=(dist > self.min_length)
            if traversable and goodlength:
                tasks['istart'].append(int(istart))
                tasks['jstart'].append(int(jstart))
                tasks['igoal'].append(int(igoal))
                tasks['jgoal'].append(int(jgoal))
                tasks['euclid'].append(float(dist))
                t += 1
        return tasks