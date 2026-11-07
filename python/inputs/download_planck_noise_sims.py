#%%
"""
Noise simulations of the Planck bands, from the Planck Legacy Archive.
======================================================================

    python download_planck_noise_sims.py [--bands 30,44,70,100,143,217,353] [--workers 8]

Downloads realizations 1..N_SIM of the full-mission PR4 noise maps,
npipe6v20_noise_<band>_mc_00200 ... 00299.fits (the only 100 there are), into
data.map_file(band, 'noise', k).

Files already on disk are skipped, so it can be rerun after an interruption;
each file is written as <name>.part and renamed once complete, and a failed
download is retried three times and then reported at the end.  --workers
files are downloaded at the same time (the PLA gives about 0.7-1.3 MB/s per
file).  The HFI maps are about 600 MB each, about 275 GB in all.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import N_SIM, PLANCK_BANDS  # noqa: E402
from data import map_file  # noqa: E402

URL = 'https://pla.esac.esa.int/pla/aio/product-action?SIMULATED_MAP.FILE_ID='


def download(dest):
    """Downloads the PLA file named as dest; True if it worked."""
    name = os.path.basename(dest)
    for attempt in range(1, 4):
        try:
            with requests.get(URL + quote(name), stream=True, timeout=600) as r:
                if r.status_code != 200:
                    raise RuntimeError(f'HTTP {r.status_code}')
                with open(dest + '.part', 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8 << 20):
                        f.write(chunk)
            os.replace(dest + '.part', dest)
            return True
        except Exception as e:
            print(f'  {name}: attempt {attempt} failed ({e})', flush=True)
            time.sleep(30)
    return False


def fetch(dest):
    t0 = time.time()
    ok = download(dest)
    if ok:
        print(f'  {os.path.basename(dest)} ({time.time() - t0:.0f} s)', flush=True)
    return dest, ok


ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
ap.add_argument('--bands', default=','.join(PLANCK_BANDS))
ap.add_argument('--workers', type=int, default=8, help='simultaneous downloads')
args = ap.parse_args()
todo = [map_file(band, 'noise', k) for band in args.bands.split(',') for k in range(1, N_SIM + 1)]
todo = [dest for dest in todo if not os.path.exists(dest)]
for dest in todo:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
print(f'{len(todo)} files to download, {args.workers} at a time', flush=True)
with ThreadPoolExecutor(max_workers=args.workers) as pool:
    failed = [os.path.basename(dest) for dest, ok in pool.map(fetch, todo) if not ok]
print(f'done; {len(failed)} failed' + (': ' + ' '.join(failed) if failed else ''), flush=True)
