#!/usr/bin/env python

echo "Get SAGAN logs on generated dataset..."

python get_logs.py A \
    --map_params "{'data_folder': './data/generated_dataset', 'maps_folder': 'maps', 'results_folder': 'results', 'results_file': 'results.csv'}" \
    --rrt_params "{'path_resolution': 1, 'step_len': 2, 'max_iter': 10000}" \
    --mu 0.1 \
    --gamma 10 \
    --n 50 \
    --output_dir logs/ \
    --output_fname sagan_generated_logs.txt \
