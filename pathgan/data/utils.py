import os
import urllib
import zipfile
from tqdm import tqdm

def download(url: str, filename: str, chunk_size: int = 4096) -> None:
    with open(filename, "wb") as fh:
        with urllib.request.urlopen(urllib.request.Request(url)) as response:
            with tqdm(total=response.length) as pbar:
                for chunk in iter(lambda: response.read(chunk_size), ""):
                    if not chunk:
                        break
                    pbar.update(chunk_size)
                    fh.write(chunk)
    print(f'Dataset downloaded!')
    return None

def extract(from_path: str, to_path: str) -> None:
    with zipfile.ZipFile(from_path, "r", compression=zipfile.ZIP_STORED) as zf:
        zf.extractall(to_path)
    print('Dataset extracted!')
    return None

def download_and_extract(root: str, url: str, filename: str=None):
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)
    if os.path.exists(fpath):
        print('Dataset is already downloaded!')
    else:
        os.makedirs(root, exist_ok=True)
        _ = download(url, fpath)
        _ = extract(fpath, root)
    return None
