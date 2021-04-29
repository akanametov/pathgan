import os
import argparse
from multiprocessing import Process

from rrt import RRT
from utils import ROIGenerator

def run_generator(roi_generator, map_name):
    roi_generator.generate(map_name=map_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = 'top', description='Run Tasks Generator')
    
    parser.add_argument('--start', type=int, default=0,
                        help='Map "from" which ROI will be obtained (default: 0)')
    
    parser.add_argument('--to', type=int, default=None,
                        help='Map "to" which ROI will be obtained (default: None)')
    
    parser.add_argument('--map_dir', default='dataset/maps',
                        help='Maps directory (default: "dataset/maps")')
    
    parser.add_argument('--task_dir', default='dataset/tasks',
                        help='Tasks directory (default: "dataset/tasks")')
    
    parser.add_argument('--save_dir', default='dataset/tasks',
                        help='Save directory (default: "dataset/tasks")')
    
    parser.add_argument('--n_runs', type=int, default=50,
                        help='Number of times searching algorithm will be runned per one map (default: 50)')
    
    args = parser.parse_args()

    print('============== ROI Generation Started ==============')
    map_names = sorted(os.listdir(args.map_dir))[args.start : args.to]

    roi_generator = ROIGenerator()
    roi_generator.set_parameters(algorithm = RRT(),
                                 n_runs = args.n_runs,
                                 map_dir = args.map_dir,
                                 task_dir = args.task_dir,
                                 save_dir = args.save_dir)
    processes = []
    for map_name in map_names:
        p = Process(target=run_generator, args=(roi_generator, map_name,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    print('===================== Finished! =====================')
