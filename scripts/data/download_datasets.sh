#!/usr/bin/env python

echo "Downloading generated dataset..."

python download.py \
    --url https://github.com/akanametov/pathgan/releases/download/2.0/dataset.zip \
    --root data/generated_dataset \

echo "Downloading movingai dataset..."

python download.py \
    --url https://github.com/akanametov/pathgan/releases/download/2.0/movingai_dataset.zip \
    --root data/movingai_dataset \
