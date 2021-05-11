import os 
from RRT_last import RRT
from multiprocessing import Process, Pool, cpu_count
from OpenAI_ROI_generator import ROIGenerator
from tqdm import tqdm

def run_generator(m_name: str):
    n_tasks = 10
    n_runs = 50
    algorithm = RRT()
    roi_generator = ROIGenerator(model=algorithm)
    roi_generator.set_parameters(m_name=m_name, 
                                 m_path='./data/dataset/OpenAI_maps/maps/', 
                                 t_path='./data/dataset/OpenAI_maps/tasks/')
    roi_generator.generate(n_runs=n_runs, n_tasks=n_tasks)


if __name__ == '__main__':
    load_dir = 'data/dataset/OpenAI_maps/maps'
    map_names = os.listdir(load_dir)
    max_ = len(map_names)
    pool = Pool(processes=cpu_count())
    for _ in tqdm(pool.imap_unordered(run_generator, map_names), total= max_):
        pass
    pool.close()
    pool.join()