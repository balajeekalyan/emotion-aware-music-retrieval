"""Extract CLAP audio embeddings with laion/larger_clap_general (replaces the defective HF laion/larger_clap_music
conversion; see phase2d_clap_zero_shot.ipynb and outputs/clap_checkpoint_validation.csv).

Protocol as in phase2b_clap_deterministic.ipynb: each excerpt is decoded at 48 kHz (first 30 s), cut into fixed,
non-overlapping 10-s windows (windows shorter than 5 s dropped), each window is embedded with the checkpoint's own
feature extractor and projection, and the track vector is the L2-normalized mean of the L2-normalized window vectors.

    python extract_clap_general.py                 # all tracks; resumes from the checkpoint
    python extract_clap_general.py --max-tracks 300   # process at most 300 new tracks, then save and exit 1 (repeat)

Writes outputs/audio_features_clap_general.parquet (Phase 2 row order: audio_clap_gen, audio_clap_gen_raw,
audio_clap_gen_windows, n_windows). Resume state: outputs/audio_features_clap_general_checkpoint.pkl.
Exit code 0 when every track is done, 1 when more remain.
"""
import argparse
import os
import pickle
import sys
import time
import warnings

import librosa
import numpy as np
import pandas as pd
import torch
from transformers import ClapModel, ClapProcessor

warnings.filterwarnings('ignore')
torch.set_num_threads(os.cpu_count())

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, 'outputs')
OUT_PATH = os.path.join(OUT_DIR, 'audio_features_clap_general.parquet')
CKPT_PATH = os.path.join(OUT_DIR, 'audio_features_clap_general_checkpoint.pkl')
CHECKPOINT = 'laion/larger_clap_general'
REVISION = None                     # filled in from the loaded config below and stored with the results
SR, WIN, MIN_WIN = 48_000, 480_000, 240_000
DURATION, BATCH_TRACKS, CKPT_EVERY = 30, 4, 200


def l2(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def windows(y):
    w = [y[i:i + WIN] for i in range(0, len(y), WIN)]
    w = [x for x in w if len(x) >= MIN_WIN]
    return w or [y]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--max-tracks', type=int, default=None)
    args = ap.parse_args()

    df = pd.read_parquet(os.path.join(OUT_DIR, 'merged_dataset.parquet'))
    audio_df = df[df['has_audio']].reset_index(drop=True)
    model = ClapModel.from_pretrained(CHECKPOINT).eval()
    proc = ClapProcessor.from_pretrained(CHECKPOINT)
    fe = proc.feature_extractor
    print(f'{CHECKPOINT}: sr={fe.sampling_rate}, max {fe.max_length_s}s, truncation={fe.truncation}, '
          f'padding={fe.padding}, fusion={model.config.audio_config.enable_fusion}', flush=True)
    assert fe.sampling_rate == SR

    def embed(wins):
        x = proc(audio=wins, sampling_rate=SR, return_tensors='pt')
        with torch.no_grad():
            return model.get_audio_features(**x).numpy().astype(np.float32)

    results = pickle.load(open(CKPT_PATH, 'rb')) if os.path.exists(CKPT_PATH) else {}
    todo = audio_df[~audio_df.song_id.isin(results)].reset_index(drop=True)
    if args.max_tracks is not None:
        todo = todo.iloc[:args.max_tracks]
    print(f'{len(results)} done, {len(todo)} to process now', flush=True)
    t0, errors = time.time(), []
    for start in range(0, len(todo), BATCH_TRACKS):
        batch = todo.iloc[start:start + BATCH_TRACKS]
        wins, owner = [], []
        for _, r in batch.iterrows():
            try:
                y = librosa.load(r.audio_path, sr=SR, mono=True, duration=DURATION)[0]
                w = windows(y); wins += w; owner += [r.song_id] * len(w)
            except Exception as e:                                   # noqa: BLE001
                errors.append((r.song_id, str(e)))
        if not wins:
            continue
        E = embed(wins); owner = np.array(owner)
        for sid in dict.fromkeys(owner):
            Ew = E[owner == sid]
            results[sid] = {'audio_clap_gen_windows': Ew, 'audio_clap_gen_raw': Ew.mean(0),
                            'audio_clap_gen': l2(l2(Ew).mean(0)), 'n_windows': len(Ew)}
        if (start // BATCH_TRACKS) % (CKPT_EVERY // BATCH_TRACKS) == 0:
            pickle.dump(results, open(CKPT_PATH, 'wb'))
            done = start + len(batch)
            print(f'  {len(results)} tracks ({done / max(time.time() - t0, 1e-6):.2f} tracks/s this run)', flush=True)
    pickle.dump(results, open(CKPT_PATH, 'wb'))
    for e in errors[:10]:
        print('  error:', e)

    if len(results) < len(audio_df):
        print(f'{len(results)} / {len(audio_df)} tracks done; run again to continue.')
        sys.exit(1)
    feat = pd.DataFrame([{'song_id': s, 'audio_clap_gen': r['audio_clap_gen'].tolist(),
                          'audio_clap_gen_raw': r['audio_clap_gen_raw'].tolist(),
                          'audio_clap_gen_windows': r['audio_clap_gen_windows'].ravel().tolist(),
                          'n_windows': r['n_windows']} for s, r in results.items()])
    feat = audio_df[['song_id']].merge(feat, on='song_id', how='left')
    X = np.vstack(feat.audio_clap_gen.values)
    assert X.shape == (len(audio_df), model.config.projection_dim) and not np.isnan(X).any()
    feat['checkpoint'] = CHECKPOINT
    feat['checkpoint_commit'] = getattr(model.config, '_commit_hash', None)
    feat.to_parquet(OUT_PATH, index=False)
    print(f'Saved {OUT_PATH}: {len(feat)} tracks, windows per track {feat.n_windows.value_counts().to_dict()}, '
          f'checkpoint commit {feat.checkpoint_commit.iloc[0]}')
    sys.exit(0)


if __name__ == '__main__':
    main()
