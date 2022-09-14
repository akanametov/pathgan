import os
import argparse
from pathgan.data.utils import download_and_extract


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "top", description="Training GAN (from original paper")
    parser.add_argument("--url", type=str, help="Url of file.")
    parser.add_argument("--root", type=str, default="data", help="Root path.")
    parser.add_argument("--filename", type=str, default=None, help="Filname.")
    args = parser.parse_args()
    os.makedirs(args.root, exist_ok=True)
    download_and_extract(args.root, args.url, args.filename)
