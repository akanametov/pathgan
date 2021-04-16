import os
import math
import pandas as pd
import numpy as np
from PIL import Image
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

def rgb2binary(img):
    return (img[..., :] > 150).astype(float)

def save_maps_and_tasks(maps, tasks, task_maps, m_path, t_path, istart=0):
    names = list(range(istart, istart+len(maps)+1))
    for i, (m, t, tms) in enumerate(zip(maps, tasks, task_maps)):
        plt.imsave(m_path+f'map_{names[i]}.png', m, cmap='gray')
        pd.DataFrame.from_dict(t).to_csv(t_path+f'map_{names[i]}.csv', index=False)
        if not os.path.exists(t_path+f'map_{names[i]}'):
            os.mkdir(t_path+f'map_{names[i]}')
        for j, tm in enumerate(tms):
            plt.imsave(t_path+f'map_{names[i]}/task_{j}.png', tm)

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
            if np.random.uniform() < self.t_prob:
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
        cstart = np.array([0, 0, 1])
        cgoal = np.array([1, 0, 0])
        tasks =  {'istart': [],
                  'jstart': [],
                  'igoal' : [],
                  'jgoal' : [],
                  'euclid': []}
        task_maps=[]
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
                task_map = np.ones((H, W, 3))
                task_map[istart, jstart] = cstart
                task_map[igoal, jgoal] = cgoal
                task_maps.append(task_map)
                t += 1
        return tasks, task_maps
    
class ROIGenerator():
    def __init__(self, model):
        self.model = model
        
    def set_parameters(self,
                       m_name=None,
                       m_path='data/dataset/maps/',
                       t_path='data/dataset/tasks/'):
        
        Map=Image.open(m_path +m_name+'.png').convert('RGB')
        Map = np.array(Map)
        self.Map = rgb2binary(Map)
        self.MapTasks = pd.read_csv(t_path +m_name+'.csv')
        self.m_name = m_name
        self.m_path = m_path
        self.t_path = t_path
        
    def generate(self, n_tasks=100, n_runs=50):
        m_name=self.m_name
        m_path=self.m_path
        t_path=self.t_path
        colors = {'roi':   np.array([0, 1, 0]),
                  'start': np.array([0, 0, 1]),
                  'goal':  np.array([1, 0, 0])}
        grids=[]
        rois=[]
        for t in tqdm(range(n_tasks)):
            start = (self.MapTasks.istart[t], self.MapTasks.jstart[t])
            goal = (self.MapTasks.igoal[t], self.MapTasks.jgoal[t])

            grid = self.Map.copy()
            roi = np.ones(grid.shape)

            for r in range(n_runs):
                path = self.model.search(self.Map[..., 1], start, goal)
                if path:
                    for x in path:
                        grid[x[0], x[1], :] = colors['roi']
                        roi[x[0], x[1], :] = colors['roi']

            grid[start[0], start[1], :] = colors['start']
            grid[goal[0], goal[1], :] = colors['goal']
            grids.append(grid)
            rois.append(roi)
            plt.imsave(t_path + m_name + f'/task_{t}_rrt.png', grid)
            plt.imsave(t_path + m_name + f'/task_{t}_roi.png', roi)
        return grids, rois