import os
import argparse

from utils import TaskGenerator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = 'top', description='Run Tasks Generator')

    parser.add_argument('--load_dir', default='dataset/maps',
                        help='Load directory (default: "dataset/maps")')
    
    parser.add_argument('--save_dir', default='dataset/tasks',
                        help='Save directory (default: "dataset/tasks")')
    
    parser.add_argument('--min_length', type=int, default=30,
                        help='Minimal Euclidian distance between "start" and "goal" points (default: 30)')
    
    parser.add_argument('--n_tasks', type=int, default=100,
                        help='Number of tasks to be generated per one map (default: 100)')
    
    args = parser.parse_args()

    print('============== Task Generation Started ==============')
    map_names = sorted(os.listdir(args.load_dir))

    task_generator = TaskGenerator()
    task_generator.set_parameters(min_length = args.min_length,
                                  n_tasks = args.n_tasks,
                                  load_dir = args.load_dir,
                                  save_dir = args.save_dir)
    
    for map_name in map_names:
        done = task_generator.generate(map_name=map_name)
        done = task_generator.save()
    print('=====================  Finished! =====================')
