"""Download the released outputs from Zenodo, verify their checksums and unpack them into outputs/.

    python download_data.py --record <ZENODO_RECORD_ID>      # e.g. the number in zenodo.org/records/<id>
    python download_data.py --zip path/to/merge_emotion_retrieval_outputs_v2.zip   # already downloaded

Existing files in outputs/ are not overwritten unless --force is given.
"""
import argparse
import hashlib
import os
import sys
import zipfile

import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
ZIP_NAME = 'merge_emotion_retrieval_outputs_v2.zip'
SUMS_NAME = 'SHA256SUMS.txt'
ZENODO_RECORD = None   # set after publishing, so `python download_data.py` works without --record


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def download(url, dest):
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        done = 0
        with open(dest + '.part', 'wb') as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f'\r  {done / 2**20:.0f} / {total / 2**20:.0f} MB', end='', flush=True)
    print()
    os.replace(dest + '.part', dest)


def fetch_from_zenodo(record, dest_dir):
    meta = requests.get(f'https://zenodo.org/api/records/{record}', timeout=60)
    meta.raise_for_status()
    files = {f['key']: f for f in meta.json()['files']}
    for name in (ZIP_NAME, SUMS_NAME):
        if name not in files:
            sys.exit(f'{name} not found in Zenodo record {record}')
        dest = os.path.join(dest_dir, name)
        expected = files[name].get('checksum', '')            # "md5:..."
        if os.path.exists(dest) and expected.startswith('md5:') and \
                hashlib.md5(open(dest, 'rb').read()).hexdigest() == expected[4:]:
            print(f'{name}: already downloaded')
            continue
        print(f'Downloading {name}')
        download(files[name]['links']['self'], dest)
    return os.path.join(dest_dir, ZIP_NAME), os.path.join(dest_dir, SUMS_NAME)


def read_sums(path):
    sums = {}
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if line and not line.startswith('#'):
            digest, name = line.split(None, 1)
            sums[name] = digest
    return sums


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--record', default=ZENODO_RECORD, help='Zenodo record id')
    ap.add_argument('--zip', help='local copy of the outputs zip (skips the download)')
    ap.add_argument('--sums', help=f'local {SUMS_NAME} (default: next to the zip)')
    ap.add_argument('--force', action='store_true', help='overwrite existing files in outputs/')
    args = ap.parse_args()

    if args.zip:
        zpath = args.zip
        spath = args.sums or os.path.join(os.path.dirname(os.path.abspath(zpath)), SUMS_NAME)
    elif args.record:
        zpath, spath = fetch_from_zenodo(args.record, ROOT)
    else:
        ap.error('give --record <id> or --zip <path>')

    sums = read_sums(spath) if os.path.exists(spath) else {}
    if not sums:
        print(f'warning: {SUMS_NAME} not found; skipping checksum verification')
    elif sums.get(os.path.basename(zpath)) not in (None, sha256(zpath)):
        sys.exit(f'checksum mismatch for {zpath}; delete it and download again')

    written = kept = 0
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            name = info.filename
            if info.is_dir():
                continue
            if not name.startswith('outputs/') or '..' in name.split('/'):
                sys.exit(f'unexpected path in zip: {name}')
            dest = os.path.join(ROOT, *name.split('/'))
            if os.path.exists(dest) and not args.force:
                kept += 1
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with z.open(info) as src, open(dest, 'wb') as out:
                while block := src.read(1 << 20):
                    out.write(block)
            if name in sums and sha256(dest) != sums[name]:
                sys.exit(f'checksum mismatch for {name}')
            written += 1
    print(f'Unpacked {written} files into outputs/' + (f'; kept {kept} existing (use --force to replace)' if kept else ''))


if __name__ == '__main__':
    main()
