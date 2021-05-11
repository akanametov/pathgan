#!/usr/bin/env python
import argparse
import yaml
import path.utils as utils

"""
Example usage:
./get_logs.py A \
--map_params "{'data_folder': './data', 'maps_folder': 'maps', 'results_folder': 'results_pix2pix', 'results_file': 'result.csv'}" \
--rrt_params "{'path_resolution': 1, 'step_len': 2, 'max_iter': 10000}" \
--mu 0.1 --gamma 10 --n 50 --output_dir ./ --output_fname pix2pix_logs.txt
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Logs generation and processing""")
    subparsers = parser.add_subparsers(help="""Choose A - to generate logs, B - to process logs""")

    parser_a = subparsers.add_parser('A', help="""Wrapper to generate RRT* logs for test maps and ROIs""")
    parser_a.add_argument('--map_params', help="""Folders containing test maps and results. 
                                                Example: {data_folder': './data', 
                                                'maps_folder': 'maps', 'results_folder': 'results_gan', 
                                                'results_file': 'result.csv'} """, type=yaml.load)
    parser_a.add_argument('--rrt_params', help="""Basic params for RRT and RRT*. 
                                                Example: {'path_resolution': 1, 
                                                'step_len': 4, 
                                                'max_iter': 5000}""", type=yaml.load)
    parser_a.add_argument('--mu', help="""Uniform sampling probability, 
                                    used in pair with ROI (ignored otherwise)""", type=float, default=0.1)
    parser_a.add_argument('--gamma', help="""Planning constant, used in n_neighbors counting for rewire. 
                                        The more gamma-the more neighbors will be considered. 
                                        Be careful with dense maps.""", type=float, default=1.)
    parser_a.add_argument('--n', help="""Number of runs for RRT* and RRT* with heuristic. 
                                    Use higher value if more accurate result is desired""", type=int)
    parser_a.add_argument('--output_dir', help="""Logs folder. It will be created, if not exists""", default='logs', type=str)
    parser_a.add_argument('--output_fname', help="""Path to output logs file""", default='logs.txt', type=str)

    parser_b = subparsers.add_parser('B', help="""All possible csv and logs generation by log file 
                                        (must be obtained earlier)""")
    parser_b.add_argument('--log_dir', type=str, help="""Root folder""")
    parser_b.add_argument('--log_file', type=str, help="""Logs from RRT*""")
    parser_b.add_argument('--collect_stats', action='store_true', help="""Merge few maps, 
                                                                if they have same type.""")

    args, _ = parser.parse_known_args()
    args_dict = vars(args)
    if 'map_params' in args_dict:
        utils.process_all_results(**args_dict)
    else:
        utils.csv_and_plots_from_logs(**args_dict)
