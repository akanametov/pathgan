import argparse
from pathgan.data.utils import download_and_extract


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "top", description="Training GAN (from original paper")
    parser.add_argument("--url", type=str, help="Url of file.")
    parser.add_argument("--root", type=str, default=".", help="Root path.")
    parser.add_argument("--filename", type=str, default=None, help="Filname.")
    args = parser.parse_args()
    download_and_extract(args.root, args.url, args.filename)
