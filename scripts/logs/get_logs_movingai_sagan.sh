#!/usr/bin/env python

echo "Get SAGAN logs on movingai dataset..."

python get_logs.py A \
    --map_params "{'data_folder': './data/movingai_dataset', 'maps_folder': 'maps', 'results_folder': 'movingai/results', 'results_file': 'result.csv'}" \
    --rrt_params "{'path_resolution': 1, 'step_len': 2, 'max_iter': 10000}" \
    --mu 0.1 \
    --gamma 10 \
    --n 50 \
    --output_dir logs/sagan \
    --output_fname sagan_movingai_logs.txt \
