"""Download the released outputs from Zenodo, verify their checksums and unpack them into outputs/.

    python download_data.py                                  # latest version of the outputs record
    python download_data.py --record <ZENODO_RECORD_ID> --exact   # exactly that version
    python download_data.py --zip path/to/merge_emotion_retrieval_outputs_v2.zip   # already downloaded
    python download_data.py --zip path/to/merge_emotion_retrieval_outputs_v2.zip.001   # parts, joined first

The Zenodo record holds the main zip as ~25 MB parts (<zip>.001, <zip>.002, ...); they are checked and joined into the
zip before it is unpacked. From version 3 of the record, a second zip (45 MB), merge_emotion_retrieval_outputs_v3_additions.zip,
holds the outputs added for the revised paper (artist-filtered results, supervised reference, coherence checks, working CLAP); it is
unpacked after the main zip. With --zip, an additions zip lying next to the main zip is used too (or pass --additions).
Existing files in outputs/ are not overwritten unless --force is given.
"""
import argparse
import glob
import hashlib
import os
import re
import sys
import zipfile

import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
ZIP_NAME = 'merge_emotion_retrieval_outputs_v2.zip'
ADDON_NAME = 'merge_emotion_retrieval_outputs_v3_additions.zip'
SUMS_NAME = 'SHA256SUMS.txt'
ZENODO_RECORD = '23275462'   # DOI 10.5281/zenodo.23275462 (v2); without --exact, the latest version is used


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


def is_part(name):
    return name.startswith(ZIP_NAME + '.') and re.fullmatch(r'\d{3}', name[len(ZIP_NAME) + 1:]) is not None


def join_parts(parts, zpath, sums):
    """Check each part against SHA256SUMS (when listed) and concatenate them, in order, into zpath."""
    for p in parts:
        name = os.path.basename(p)
        if name in sums and sha256(p) != sums[name]:
            sys.exit(f'checksum mismatch for {name}; delete it and download again')
    print(f'Joining {len(parts)} parts into {os.path.basename(zpath)}')
    with open(zpath + '.tmp', 'wb') as out:
        for p in parts:
            with open(p, 'rb') as f:
                while block := f.read(1 << 20):
                    out.write(block)
    os.replace(zpath + '.tmp', zpath)


def fetch_from_zenodo(record, dest_dir, exact):
    url = f'https://zenodo.org/api/records/{record}' + ('' if exact else '/versions/latest')
    meta = requests.get(url, timeout=60)
    meta.raise_for_status()
    meta = meta.json()
    print(f'Zenodo record {meta["id"]}' + ('' if exact else f' (latest version of {record})'))
    files = {f['key']: f for f in meta['files']}
    parts = [] if ZIP_NAME in files else sorted(k for k in files if is_part(k))
    wanted = [SUMS_NAME] + (parts or [ZIP_NAME]) + ([ADDON_NAME] if ADDON_NAME in files else [])
    for name in wanted:
        if name not in files:
            sys.exit(f'{name} not found in Zenodo record {meta["id"]}')
        dest = os.path.join(dest_dir, name)
        expected = files[name].get('checksum', '')            # "md5:..."
        if os.path.exists(dest) and expected.startswith('md5:') and \
                hashlib.md5(open(dest, 'rb').read()).hexdigest() == expected[4:]:
            print(f'{name}: already downloaded')
            continue
        print(f'Downloading {name}')
        download(files[name]['links']['self'], dest)
    zpath, spath = os.path.join(dest_dir, ZIP_NAME), os.path.join(dest_dir, SUMS_NAME)
    if parts:
        join_parts([os.path.join(dest_dir, p) for p in parts], zpath, read_sums(spath))
    apath = os.path.join(dest_dir, ADDON_NAME) if ADDON_NAME in files else None
    return zpath, spath, apath


def read_sums(path):
    sums = {}
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if line and not line.startswith('#'):
            digest, name = line.split(None, 1)
            sums[name] = digest
    return sums


def unpack(zpath, sums, force):
    """Unpack one zip into outputs/, checking every file listed in SHA256SUMS."""
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
            if os.path.exists(dest) and not force:
                kept += 1
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with z.open(info) as src, open(dest, 'wb') as out:
                while block := src.read(1 << 20):
                    out.write(block)
            if name in sums and sha256(dest) != sums[name]:
                sys.exit(f'checksum mismatch for {name}')
            written += 1
    print(f'{os.path.basename(zpath)}: unpacked {written} files into outputs/'
          + (f'; kept {kept} existing (use --force to replace)' if kept else ''))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--record', default=ZENODO_RECORD, help='Zenodo record id')
    ap.add_argument('--exact', action='store_true', help='use exactly --record, not its latest version')
    ap.add_argument('--zip', help='local copy of the outputs zip (skips the download)')
    ap.add_argument('--additions', help=f'local copy of {ADDON_NAME} (default: next to --zip, if present)')
    ap.add_argument('--sums', help=f'local {SUMS_NAME} (default: next to the zip)')
    ap.add_argument('--force', action='store_true', help='overwrite existing files in outputs/')
    args = ap.parse_args()

    if args.zip:
        zpath = args.zip
        folder = os.path.dirname(os.path.abspath(zpath))
        spath = args.sums or os.path.join(folder, SUMS_NAME)
        apath = args.additions or (os.path.join(folder, ADDON_NAME)
                                   if os.path.exists(os.path.join(folder, ADDON_NAME)) else None)
    elif args.record:
        zpath, spath, apath = fetch_from_zenodo(args.record, ROOT, args.exact)
    else:
        ap.error('give --record <id> or --zip <path>')

    sums = read_sums(spath) if os.path.exists(spath) else {}
    if is_part(os.path.basename(zpath)):
        base = zpath[:-4]
        parts = sorted(p for p in glob.glob(glob.escape(base) + '.*') if is_part(os.path.basename(p)))
        join_parts(parts, base, sums)
        zpath = base
    unpack(zpath, sums, args.force)
    if apath:
        unpack(apath, sums, args.force)


if __name__ == '__main__':
    main()
