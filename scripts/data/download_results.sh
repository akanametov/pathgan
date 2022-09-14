#!/usr/bin/env python

echo "Downloading SAGAN results on generated dataset..."

python download.py \
    --url https://github.com/akanametov/pathgan/releases/download/2.0/results.zip \
    --root data/generated_dataset \

echo "Downloading SAGAN results on movingai dataset..."

python download.py \
    --url https://github.com/akanametov/pathgan/releases/download/2.0/movingai_results.zip \
    --root data/movingai_dataset/movingai \

echo "Downloading Pix2pix results on generated dataset..."

python download.py \
    --url https://github.com/akanametov/pathgan/releases/download/2.0/pixresults.zip \
    --root data/generated_dataset \

echo "Downloading Pix2pix results on movingai dataset..."

python download.py \
    --url https://github.com/akanametov/pathgan/releases/download/2.0/movingai_pixresults.zip \
    --root data/movingai_dataset/movingai \
