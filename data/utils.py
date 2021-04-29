import os
import math
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from progress.bar import IncrementalBar

def rgb2binary(img):
    return (img[..., :] > 150).astype(float)

class MapAugmentator():
    def __init__(self,):
        pass
        
    def set_parameters(self,
                       height_shift=2,
                       width_shift=2,
                       shift_step=1,
                       rot_prob=0.5,
                       n_maps=10,
                       load_dir='data/dataset/init_maps',
                       save_dir='data/dataset/maps'):
        
        self.h_shift = height_shift
        self.w_shift = width_shift
        self.h_range = np.arange(- height_shift, height_shift, shift_step)
        self.w_range = np.arange(- width_shift, width_shift, shift_step)
        self.step = shift_step
        self.t_prob = rot_prob
        self.n_maps = n_maps
        self.load_dir = load_dir
        self.save_dir = save_dir
        
    def augment(self, map_name):
        self.map_name = map_name.split('.')[0]
        map_img = Image.open(f'{self.load_dir}/{map_name}')
        map_data = rgb2binary(np.array(map_img))[..., 0]
        
        maps = []
        H, W = map_data.shape
        nH = H + 2*self.h_shift
        nW = W + 2*self.w_shift
        bar = IncrementalBar(f'{map_name}:', max=self.n_maps)
        for n in range(self.n_maps):
            grid = np.ones((nH, nW))
            
            h = np.random.choice(self.h_range, 1).item()
            w = np.random.choice(self.w_range, 1).item()
            
            grid[self.h_shift + self.step*h: H + self.h_shift + self.step*h,
                 self.w_shift + self.step*w: W + self.w_shift + self.step*w] = map_data
            g_map = grid[self.h_shift: - self.h_shift, self.w_shift: - self.w_shift].copy()
            if np.random.uniform() < self.t_prob:
                g_map = g_map.T
            assert g_map.shape == (H, W)
            maps.append(g_map)
            bar.next()
        bar.finish()
        self.aug_maps=maps
        return True
    
    def save(self,):
        if not os.path.exists(f'{self.save_dir}'):
            os.mkdir(f'{self.save_dir}')
        for i, aug_map in enumerate(self.aug_maps):
            save_path = f'{self.save_dir}/{self.map_name}{i}.png'
            plt.imsave(save_path, aug_map, cmap='gray')
        return True
        
    
class TaskGenerator():
    def __init__(self,):
        pass
        
    def set_parameters(self,
                       min_length=30,
                       n_tasks=100,
                       load_dir='data/dataset/maps',
                       save_dir='data/dataset/tasks'):
        
        self.min_length = min_length
        self.n_tasks = n_tasks
        self.load_dir = load_dir
        self.save_dir = save_dir
        
    def euclid(self, i1, j1, i2, j2):
        return math.sqrt((i1 - i2)**2 + (j1 - j2)**2)
        
    def generate(self, map_name):
        self.map_name = map_name.split('.')[0]
        map_img = Image.open(f'{self.load_dir}/{map_name}')
        map_data = rgb2binary(np.array(map_img))[..., 0]
        start_color = np.array([0, 0, 1])
        goal_color = np.array([1, 0, 0])
        tasks_data = {'istart': [], 'jstart': [],
                      'igoal' : [], 'jgoal' : [],
                      'euclid': []}
        task_maps=[]
        H, W = map_data.shape
        t=0
        bar = IncrementalBar(f'{map_name}:', max=self.n_tasks)
        while t < self.n_tasks:
            istart = np.random.choice(H, 1).item()
            jstart = np.random.choice(W, 1).item()
            igoal = np.random.choice(H, 1).item()
            jgoal = np.random.choice(W, 1).item()
            traversable = (map_data[istart, jstart] != 0) and (map_data[igoal, jgoal] != 0)
            dist = self.euclid(istart, jstart, igoal, jgoal)
            goodlength = (dist > self.min_length)
            if traversable and goodlength:
                tasks_data['istart'].append(int(istart))
                tasks_data['jstart'].append(int(jstart))
                tasks_data['igoal'].append(int(igoal))
                tasks_data['jgoal'].append(int(jgoal))
                tasks_data['euclid'].append(float(dist))
                task_map = np.ones((H, W, 3))
                task_map[istart, jstart] = start_color
                task_map[igoal, jgoal] = goal_color
                task_maps.append(task_map)
                t += 1
                bar.next()
        bar.finish()
        self.tasks_data = tasks_data
        self.task_maps = task_maps
        return True
    
    def save(self,):
        if not os.path.exists(f'{self.save_dir}'):
            os.mkdir(f'{self.save_dir}')
        csv_file = pd.DataFrame.from_dict(self.tasks_data)
        fname = f'{self.save_dir}/{self.map_name}.csv'
        csv_file.to_csv(fname, index=False)
        if not os.path.exists(f'{self.save_dir}/{self.map_name}'):
            os.mkdir(f'{self.save_dir}/{self.map_name}')
        for i, task_map in enumerate(self.task_maps):
            save_path = f'{self.save_dir}/{self.map_name}/task_{i}.png'
            plt.imsave(save_path, task_map)
        return True
    
class ROIGenerator():
    def __init__(self,):
        pass
    
    def set_parameters(self,
                       algorithm = None,
                       n_runs = 50,
                       map_dir = 'data/dataset/maps',
                       task_dir = 'data/dataset/tasks',
                       save_dir = 'data/dataset/tasks'):
        self.algorithm = algorithm
        self.n_runs = n_runs
        self.map_dir = map_dir
        self.task_dir = task_dir
        self.save_dir = save_dir
        
    def generate(self, map_name):
        self.map_name = map_name.split('.')[0]
        if not os.path.exists(f'{self.save_dir}'):
            os.mkdir(f'{self.save_dir}')
        if not os.path.exists(f'{self.save_dir}/{self.map_name}'):
            os.mkdir(f'{self.save_dir}/{self.map_name}')
        map_img = Image.open(f'{self.map_dir}/{map_name}').convert('RGB')
        map_data = rgb2binary(np.array(map_img))
        csv_file = pd.read_csv(f'{self.task_dir}/{self.map_name}.csv')
        
        colors = {'roi':   np.array([0, 1, 0]),
                  'start': np.array([0, 0, 1]),
                  'goal':  np.array([1, 0, 0])}
        grids=[]
        rois=[]
        bar = IncrementalBar(f'{map_name}:', max=len(csv_file))
        for i, row in csv_file.iterrows():
            start = (int(row.istart), int(row.jstart))
            goal = (int(row.igoal), int(row.jgoal))
            grid = map_data.copy()
            roi = np.ones(grid.shape)
            for _ in range(self.n_runs):
                path = self.algorithm.run(map_data[..., 1], start, goal)
                if path:
                    for x in path:
                        grid[x[0], x[1], :] = colors['roi']
                        roi[x[0], x[1], :] = colors['roi']
            grid[start[0], start[1], :] = colors['start']
            grid[goal[0], goal[1], :] = colors['goal']
            res_path = f'{self.save_dir}/{self.map_name}/task_{i}_rrt.png'
            roi_path = f'{self.save_dir}/{self.map_name}/task_{i}_roi.png'
            plt.imsave(res_path, grid)
            plt.imsave(roi_path, roi)
            bar.next()
        bar.finish()