from rrt import RRT
from data.utils import ROIGenerator
from multiprocessing import Process
import argparse


def run_generator(m_name: str):
    n_tasks = 100
    n_runs = 50
    algorithm = RRT()
    roi_generator = ROIGenerator(model=algorithm)
    roi_generator.set_parameters(m_name=m_name, 
                                 m_path='data/dataset/maps/', 
                                 t_path='data/dataset/tasks/')
    roi_generator.generate(n_runs=n_runs, n_tasks=n_tasks)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, help='Start map number')
    parser.add_argument('--end', type=int, help='End map number (not including)')
    args, _ = parser.parse_known_args()
    
    processes = []
    start, end = args.start, args.end
    for n in range(start, end):
        m_name = '_'.join(['map', str(n)])
        p = Process(target=run_generator, args=(m_name,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()