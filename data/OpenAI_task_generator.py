import os 
import numpy as np
import pandas as pd
from PIL import Image
from data.utils import rgb2binary, TaskGenerator
from pathlib import Path
import argparse
from pprint import pprint
import matplotlib.pyplot as plt
from progress.bar import IncrementalBar


class TaskGenerator_updated(TaskGenerator):

    def __init__(self, config):
        self.from_kwargs(config)

        self.H, self.W = None, None
        self.cstart = np.array([0, 0, 1])
        self.cgoal = np.array([1, 0, 0])

    def from_kwargs(self, config_):
        for (field, value) in config_.items():
            setattr(self, field, value)
        return self
    
    def set_gmap(self, gmap):
        setattr(self, 'gmap', gmap)
        self.H, self.W = self.gmap.shape
        return self
    
    def generate(self):
        tasks =  {'istart': [], 'jstart': [], 'igoal' : [], 'jgoal' : [], 'euclid': []}
        task_maps=[]
        t=0
        while t < self.n_tasks:
            istart = np.random.choice(self.H, 1).item()
            jstart = np.random.choice(self.W, 1).item()
            igoal = np.random.choice(self.H, 1).item()
            jgoal = np.random.choice(self.W, 1).item()
            traversable=(self.gmap[istart, jstart] != 0) and (self.gmap[igoal, jgoal] != 0)
            dist = self.euclid(istart, jstart, igoal, jgoal)
            goodlength=(dist > self.min_length)
            if traversable and goodlength:
                tasks['istart'].append(int(istart))
                tasks['jstart'].append(int(jstart))
                tasks['igoal'].append(int(igoal))
                tasks['jgoal'].append(int(jgoal))
                tasks['euclid'].append(float(dist))
                task_map = np.ones((self.H, self.W, 3))
                task_map[istart, jstart] = self.cstart
                task_map[igoal, jgoal] = self.cgoal
                task_maps.append(task_map)
                t += 1
        return tasks, task_maps

    def save(self, map_file_name, tasks, task_maps):
        
        if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)
        
        current_path = self.save_dir + f'/{map_file_name}'
        
        if not os.path.exists(current_path):
                os.mkdir(current_path)    
                
        pd.DataFrame.from_dict(tasks).to_csv(current_path + '.csv', index=False)
        
        for i, tm in enumerate(task_maps):
            plt.imsave(current_path + f'/task_{i}.png', tm)    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog = 'top', description='Run Tasks Generator')

    parser.add_argument('--load_dir', default='data/dataset/maps',
                        help='Load directory (default: "data/dataset/maps")')
    parser.add_argument('--save_dir', default='data/dataset/tasks',
                        help='Save directory (default: "data/dataset/tasks")')
    parser.add_argument('--min_length', type=int, default=30,
                        help='Minimal Euclidian distance between "start" and "goal" points (default: 30)')
    parser.add_argument('--n_tasks', type=int, default=100,
                        help='Number of tasks to be generated per one map (default: 100)')
    
    args = parser.parse_args()

    print('============== Tasks Generation Started ==============')
    
    task_generator = TaskGenerator_updated(vars(args))
    map_names = os.listdir(task_generator.load_dir)
      
    bar = IncrementalBar('Maps:', max=len(map_names))
    for map_name in map_names:
        fname = Path(map_name).stem
        init_map = Image.open(args.load_dir + map_name).convert("RGB")
        init_map = rgb2binary(np.array(init_map))
        task_generator.set_gmap(init_map[..., 0])
        tasks, tasks_maps = task_generator.generate()
        task_generator.save(fname, tasks, tasks_maps)

        # done = task_generator.generate(map_name=map_name)
        # done = task_generator.save()

        bar.next()
    bar.finish()
    print('=====================  Finished! =====================')