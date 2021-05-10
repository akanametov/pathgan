import os
import re
import numpy as np
import pandas as pd

from typing import Tuple, List, Union, Optional, Iterable
from collections import defaultdict, OrderedDict
from PIL import Image
from tqdm import tqdm

from functools import reduce
import operator
import multiprocessing
import json

from itertools import product

import path
from path.rrt_base import RRTBase, PathDescription
from path.rrt_star import RRTStar


class Report(object):
    def __init__(self, map_name: str):
        self.map_name = map_name
        self.first = defaultdict(list)
        self.best = defaultdict(list)
        self.costs = []
        self.path_lengths = []
        self.pad_len = 0
        self.samples_taken = []
        self.nodes_taken = []

    def update(self,
               first_path: PathDescription,
               best_path: PathDescription,
               costs: List[float], paths: List[int],
               samples: List[int], nodes: List[int]):
        best_dict = best_path()
        for ct_name, ct_value in first_path().items():
            if ct_name != 'path':
                self.first[ct_name].append(ct_value)
                self.best[ct_name].append(best_dict[ct_name])
        self.pad_len = max(self.pad_len, len(costs))
        self.costs.append(costs)
        self.path_lengths.append(paths)
        self.samples_taken.append(samples)
        self.nodes_taken.append(nodes)

    @staticmethod
    def get_mean_std(a: np.ndarray, axis: int = 0):
        mu = np.nanmean(a, axis=axis)
        std = np.nanstd(a, axis=axis)
        return mu.tolist(), std.tolist()

    def __call__(self):
        for i in range(len(self.costs)):
            l = self.pad_len - len(self.costs[i])
            if l > 0:
                self.costs[i].extend([np.nan] * l)
                self.path_lengths[i].extend([np.nan] * l)

        if not isinstance(self.costs, np.ndarray):
            self.costs = np.vstack(self.costs)
            self.path_lengths = np.vstack(self.path_lengths)
            self.samples_taken = np.array(self.samples_taken)
            self.nodes_taken = np.array(self.nodes_taken)

        mean_costs, std_costs = self.get_mean_std(self.costs)
        mean_paths, std_paths = self.get_mean_std(self.path_lengths)
        mean_samples, std_samples = self.get_mean_std(self.samples_taken)
        mean_nodes, std_nodes = self.get_mean_std(self.nodes_taken)

        report_dict = {'map_name': self.map_name,
                       'first': self.first,
                       'best': self.best,
                       'costs': {'mean': mean_costs, 'std': std_costs},
                       'paths': {'mean': mean_paths, 'std': std_paths},
                       'samples': {'mean': mean_samples, 'std': std_samples},
                       'nodes': {'mean': mean_nodes, 'std': std_nodes}
                       }
        return report_dict


def get_n_results(data_folder: str = '../data',
                  results_folder: str = 'results',
                  results_file: str = 'result.csv') -> int:
    results_file = os.path.join(data_folder, results_folder, results_file)
    roi_description = pd.read_csv(results_file, header=None)
    return roi_description.shape[0] - 1


def process_all_results(map_params: dict,
                        rrt_params: dict,
                        mu: float = 0.1,
                        gamma: float = 10.,
                        n: int = 50,
                        output_dir: str = 'logs',
                        output_fname: str = 'logs.txt'):
    seen_maps = set()
    n_results = get_n_results(map_params['data_folder'],
                              map_params['results_folder'],
                              map_params['results_file'])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, output_fname)
    with tqdm(total=n_results) as pbar:
        with open(output_path, 'w') as f:
            for i in range(n_results):
                map_params['result_row_id'] = i
                out = get_map_and_task(**map_params)
                map_name = out['grid_map'].split('/')[-1]
                if map_name in seen_maps:
                    continue

                pbar.write('Processing map {}...'.format(map_name))
                seen_maps.add(map_name)

                data = {
                    'grid_map': process_image(out['grid_map']),
                    'xy_init': out['xy_init'],
                    'xy_goal': out['xy_goal'],
                    'dist_init_goal': out['euclid']
                }
                roi_data = {'roi': roi_from_image(out['pred_roi']),
                            'mu': mu}
                rewire_params = {'gamma': gamma}
                report1 = run_experiment(map_name, RRTStar,
                                         {**data, **rrt_params, **rewire_params}, n)
                report2 = run_experiment(map_name, RRTStar,
                                         {**data, **rrt_params, **roi_data, **rewire_params}, n)
                f.write(json.dumps(report1))
                f.write('\n')
                f.write(json.dumps(report2))
                f.write('\n')
                pbar.update(1)


def get_map_and_task(data_folder: str = '../data',
                     maps_folder: str = 'maps',
                     results_folder: str = 'results',
                     results_file: str = 'result.csv',
                     result_row_id: int = 0) -> dict:
    results_file = os.path.join(data_folder, results_folder, results_file)
    roi_description = pd.read_csv(results_file, header=None,
                                  skiprows=result_row_id + 1, nrows=1)
    true_roi = roi_description.iloc[0, 0]
    pred_roi = re.split('[\\\\/]', roi_description.iloc[0, 1])
    if len(pred_roi) == 3:
        pred_roi = pred_roi[2]
    else:
        pred_roi = pred_roi[1]
    dataset_folder, tasks_folder, map_name, task_roi_name = re.split('[\\\\/]', true_roi)
    map_path = os.path.join(data_folder,
                            dataset_folder,
                            maps_folder,
                            map_name + '.png')
    task_path = os.path.join(data_folder,
                             dataset_folder,
                             tasks_folder,
                             map_name + '.csv')
    task_idx = int(task_roi_name.split('_')[1])
    task_description = pd.read_csv(task_path, header=None,
                                   skiprows=task_idx + 1, nrows=1).values.tolist()[0]
    x0, y0, x1, y1 = list(map(int, task_description[:-1]))
    euclid = task_description[-1]

    true_roi_path = os.path.join(data_folder, dataset_folder,
                                 tasks_folder, map_name, task_roi_name)
    pred_roi_path = os.path.join(data_folder, results_folder, pred_roi)

    return {'grid_map': map_path,
            'true_roi': true_roi_path,
            'pred_roi': pred_roi_path,
            'xy_init': (x0, y0),
            'xy_goal': (x1, y1),
            'euclid': euclid}


def rgb2binary(img: np.ndarray) -> np.ndarray:
    return (img[..., :] > 150).astype(float)


def process_image(load_dir: str) -> np.ndarray:
    img = Image.open(load_dir).convert('RGB')
    data = rgb2binary(np.array(img))
    return data


def roi_from_image(load_dir: str) -> List[Tuple[int, int]]:
    roi_data = process_image(load_dir)
    mask = roi_data[..., 0] * roi_data[..., 2]
    roi = list(zip(*np.where(mask == 0)))
    return roi


def wrapper(algo: RRTBase, proc_num: int, return_dict: dict):
    algo.run()
    return_dict[proc_num] = algo


def run_experiment(map_name: str, algorithm: RRTBase, params: dict, n: int = 50):
    report = Report(map_name)
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for seed in range(n):
        params['seed'] = seed
        algo = algorithm(**params)
        p = multiprocessing.Process(target=wrapper, args=(algo, seed, return_dict))
        jobs.append(p)
        p.start()
    with tqdm(total=n) as pbar:
        for p in jobs:
            p.join()
            pbar.update(1)
    for algo in return_dict.values():
        report.update(algo.first_path,
                      algo.best_path,
                      algo.costs_history,
                      algo.path_lengths_history,
                      algo.samples_taken_history,
                      algo.nodes_taken_history)
    return report()


def get_by_keys(log_dict: dict,
                keys: list) -> Union[Iterable, dict]:
    return reduce(operator.getitem, keys, log_dict)


def set_by_keys(log_dict: dict,
                keys: list):
    value = get_by_keys(log_dict, keys[:-1])[keys[-1]]
    get_by_keys(log_dict, keys[:-1])[keys[-1]] = [value]


def rename(name: str, pattern: str):
    pattern = re.compile(pattern)
    num = re.findall(pattern, name)[0]
    name = name.replace(num, str(int(num) // 10)).split('.')[0]
    return name


def collect_stats_by_maps(log_dir: str,
                          log_file: str,
                          pattern: Optional[str] = None):
    seen_maps = set()
    log_dict_rrt_s = OrderedDict()
    log_dict_rrt_s_h = OrderedDict()
    log_path = os.path.join(log_dir, log_file)
    with open(log_path) as f:
        while True:
            rrt_s_line = f.readline()
            if not rrt_s_line:
                break
            rrt_s_dict = json.loads(rrt_s_line.rstrip('\n'))
            rrt_s_h_dict = json.loads(f.readline().rstrip('\n'))
            map_name = rrt_s_dict['map_name']
            if pattern is not None:
                map_name = rename(map_name, pattern)
            del rrt_s_dict['map_name']
            del rrt_s_h_dict['map_name']
            if map_name not in seen_maps:
                seen_maps.add(map_name)
                for key in rrt_s_dict:
                    for key_ in rrt_s_dict[key]:
                        rrt_s_dict[key][key_] = [rrt_s_dict[key][key_]]
                        rrt_s_h_dict[key][key_] = [rrt_s_h_dict[key][key_]]
                log_dict_rrt_s[map_name] = rrt_s_dict
                log_dict_rrt_s_h[map_name] = rrt_s_h_dict
            else:
                for key in rrt_s_dict:
                    for key_ in rrt_s_dict[key]:
                        log_dict_rrt_s[map_name][key][key_].append(rrt_s_dict[key][key_])
                        log_dict_rrt_s_h[map_name][key][key_].append(rrt_s_h_dict[key][key_])
    for map_name in log_dict_rrt_s:
        for key in log_dict_rrt_s[map_name]:
            for key_ in log_dict_rrt_s[map_name][key]:
                # list of lists, get mean
                counts = log_dict_rrt_s[map_name][key][key_]
                counts_h = log_dict_rrt_s_h[map_name][key][key_]
                # for padding
                l = max(map(lambda x: len(x), counts))
                l_h = max(map(lambda x: len(x), counts_h))
                for i, lst in enumerate(counts):
                    cur_l = l - len(lst)
                    if cur_l > 0:
                        counts[i] = lst + [np.nan] * cur_l

                for i, lst in enumerate(counts_h):
                    cur_l = l_h - len(lst)
                    if cur_l > 0:
                        counts_h[i] = lst + [np.nan] * cur_l
                counts = np.array(np.vstack(counts)).astype(float)
                counts_h = np.array(np.vstack(counts_h)).astype(float)
                counts[counts == float('inf')] = np.nan
                counts_h[counts_h == float('inf')] = np.nan

                counts = np.nanmean(counts, axis=0).tolist()
                counts_h = np.nanmean(counts_h, axis=0).tolist()
                log_dict_rrt_s[map_name][key][key_] = counts
                log_dict_rrt_s_h[map_name][key][key_] = counts_h
    output_filename = os.path.join(log_dir, '_'.join(['collected_stats', log_file]))
    with open(output_filename, 'w') as f:
        for map_name in log_dict_rrt_s:
            map_dict = {'map_name': map_name}
            rrt_s_dict = {**map_dict, **log_dict_rrt_s[map_name]}
            f.write(json.dumps(rrt_s_dict))
            f.write('\n')
            rrt_s_h_dict = {**map_dict, **log_dict_rrt_s_h[map_name]}
            f.write(json.dumps(rrt_s_h_dict))
            f.write('\n')
    return output_filename


def get_stats_table(map_name: str,
                    pad_len: int,
                    counts: list):
    map_name = [map_name.split('.')[0]] * 2 * pad_len
    algo_type = ['RRT*-uniform'] * pad_len + ['RRT*-ROI'] * pad_len
    return np.vstack((map_name, algo_type, counts))


def get_stats_table_by_keys(log_dir: str,
                            log_file: str,
                            keys: List[str],
                            csv_name: str):
    tables = []
    log_path = os.path.join(log_dir, log_file)
    with open(log_path) as f:
        while True:
            rrt_s_line = f.readline()
            if not rrt_s_line:
                break
            rrt_s = json.loads(rrt_s_line.rstrip('\n'))
            rrt_s_h = json.loads(f.readline().rstrip('\n'))

            stats_s = get_by_keys(rrt_s, keys)
            stats_s_h = get_by_keys(rrt_s_h, keys)
            map_name = rrt_s['map_name'].split('.')[0]
            table = get_stats_table(map_name, len(stats_s), stats_s + stats_s_h)
            tables.append(table)
    tables = np.hstack(tables).T
    df = pd.DataFrame(tables, columns=['map_name', 'algo_type', 'counts'])
    df[['counts']] = df[['counts']].astype(float)
    df.to_csv(os.path.join(log_dir, csv_name))
    return df


def get_plot_data_by_keys(log_dir: str,
                          log_file: str,
                          keys: List[str]) -> dict:
    plot_data = {}
    log_path = os.path.join(log_dir, log_file)
    with open(log_path) as f:
        while True:
            rrt_s_line = f.readline()
            if not rrt_s_line:
                break
            rrt_s = json.loads(rrt_s_line.rstrip('\n'))
            rrt_s_h = json.loads(f.readline().rstrip('\n'))
            data = get_by_keys(rrt_s, keys)
            data_h = get_by_keys(rrt_s_h, keys)

            map_name = rrt_s['map_name'].split('.')[0]
            plot_data[map_name] = {'uniform': data, 'roi': data_h}
    return plot_data


def csv_and_plots_from_logs(log_dir: str,
                            log_file: str,
                            collect_stats: bool = False,
                            pattern: str = '\d+'):
    if collect_stats:
        _, log_file = os.path.split(collect_stats_by_maps(log_dir, log_file, pattern))
    prefix = '_'.join(log_file.split('_')[:-1])
    for path_type, metric_name in product(path.PATH_TYPES, path.PATH_METRICS_KEYS):
        csv_name = f'{prefix}_{path_type}_{metric_name}.csv'
        get_stats_table_by_keys(log_dir, log_file, [path_type, metric_name], csv_name)
    for metric_name in path.RUNS_METRICS:
        plot_data = get_plot_data_by_keys(log_dir, log_file, [metric_name])
        plot_file = f'{prefix}_{metric_name}.plot'
        output = os.path.join(log_dir, plot_file)
        with open(output, 'w') as f:
            f.write(json.dumps(plot_data))
            f.write('\n')