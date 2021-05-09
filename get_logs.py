#!/usr/bin/env python
import argparse
import yaml
from path.utils import process_all_results

"""
Example usage:
./get_logs.py \
--map_params "{'data_folder': './data', 'maps_folder': 'maps', 'results_folder': 'results_gan', 'results_file': 'result.csv'}" \
--rrt_params "{'path_resolution': 1, 'step_len': 4, 'max_iter': 100}" \
--mu 0.1 --gamma 100 --n 50 --output_fname gan_logs.txt
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wrapper to generate RRT* logs for test maps and ROIs')
    parser.add_argument('--map_params', help="""Folders containing test maps and results. 
                                                Example: {data_folder': './data', 
                                                'maps_folder': 'maps', 'results_folder': 'results_gan', 
                                                'results_file': 'result.csv'} """, type=yaml.load)
    parser.add_argument('--rrt_params', help="""Basic params for RRT and RRT*. 
                                                Example: {'path_resolution': 1, 
                                                'step_len': 4, 
                                                'max_iter': 5000}""", type=yaml.load)
    parser.add_argument('--mu', help="""Uniform sampling probability, 
                                    used in pair with ROI (ignored otherwise)""", type=float, default=0.1)
    parser.add_argument('--gamma', help="""Planning constant, used in n_neighbors counting for rewire. 
                                        The more gamma-the more neighbors will be considered. 
                                        Be careful with dense maps.""", type=float, default=1.)
    parser.add_argument('--n', help="""Number of runs for RRT* and RRT* with heuristic. 
                                    Use higher value if more accurate result is desired""", type=int)
    parser.add_argument('--output_fname', help="""Path to output logs file""", default='logs.txt', type=str)

    args, _ = parser.parse_known_args()
    process_all_results(**vars(args))