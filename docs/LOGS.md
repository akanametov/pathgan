Logs generation and processing

```
./pathgan> ./get_logs.py --help
usage: get_logs.py [-h] {A,B} ...

Logs generation and processing

positional arguments:
  {A,B}       Choose A - to generate logs, B - to process logs
    A         Wrapper to generate RRT* logs for test maps and ROIs
    B         All possible csv and logs generation by log file (must be
              obtained earlier)

optional arguments:
  -h, --help  show this help message and exit

```

```
./pathgan>./get_logs.py A --help
usage: get_logs.py A [-h] [--map_params MAP_PARAMS] [--rrt_params RRT_PARAMS]
                     [--mu MU] [--gamma GAMMA] [--n N]
                     [--output_dir OUTPUT_DIR] [--output_fname OUTPUT_FNAME]

optional arguments:
  -h, --help            show this help message and exit
  --map_params MAP_PARAMS
                        Folders containing test maps and results. Example:
                        {data_folder': './data', 'maps_folder': 'maps',
                        'results_folder': 'results_gan', 'results_file':
                        'result.csv'}
  --rrt_params RRT_PARAMS
                        Basic params for RRT and RRT*. Example:
                        {'path_resolution': 1, 'step_len': 4, 'max_iter':
                        5000}
  --mu MU               Uniform sampling probability, used in pair with ROI
                        (ignored otherwise)
  --gamma GAMMA         Planning constant, used in n_neighbors counting for
                        rewire. The more gamma-the more neighbors will be
                        considered. Be careful with dense maps.
  --n N                 Number of runs for RRT* and RRT* with heuristic. Use
                        higher value if more accurate result is desired
  --output_dir OUTPUT_DIR
                        Logs folder. It will be created, if not exists
  --output_fname OUTPUT_FNAME
                        Path to output logs file
```

```
./pathgan> ./get_logs.py B --help
usage: get_logs.py B [-h] [--log_dir LOG_DIR] [--log_file LOG_FILE]
                     [--collect_stats]

optional arguments:
  -h, --help           show this help message and exit
  --log_dir LOG_DIR    Root folder
  --log_file LOG_FILE  Logs from RRT*
  --collect_stats      Merge few maps, if they have same type.

```
