"""Memory-bounded MERT / MuQ extraction (same protocol as phase2c_mert_muq.ipynb).

Processes at most --max-tracks tracks per invocation, checkpointing every 50, then exits so a
driver loop can start a fresh process. Exit code 0 = this model is complete, 3 = more to do.

    python extract_audio_encoders.py mert --max-tracks 300
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import librosa
import torch

warnings.filterwarnings('ignore')

ROOT    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, 'outputs')
ENC_DIR = os.path.join(OUT_DIR, 'encoders')
SR, WIN, MIN_WIN, DURATION, CKPT_EVERY = 24_000, 240_000, 120_000, 30, 50
MERT_REV = '12af15fef9d0ac838c3f475bfbbf26d2060dd4f5'
DIMS = {'mert': 768, 'muq': 1024}


def windows(y):
    w = [y[i:i + WIN] for i in range(0, len(y), WIN)]
    w = [x for x in w if len(x) >= MIN_WIN]
    return w or [y]


def load_model(name):
    if name == 'mert':
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        m = AutoModel.from_pretrained('m-a-p/MERT-v1-95M', trust_remote_code=True, revision=MERT_REV).eval()
        proc = Wav2Vec2FeatureExtractor.from_pretrained('m-a-p/MERT-v1-95M', trust_remote_code=True,
                                                        revision=MERT_REV)
        return lambda wins: m(**proc(wins, sampling_rate=SR, return_tensors='pt'),
                              output_hidden_states=True).hidden_states
    from muq import MuQ
    m = MuQ.from_pretrained('OpenMuQ/MuQ-large-msd-iter').eval()
    return lambda wins: m(torch.tensor(np.stack(wins)), output_hidden_states=True).hidden_states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model', choices=['mert', 'muq'])
    ap.add_argument('--max-tracks', type=int, default=300)
    ap.add_argument('--threads', type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    name, dim = args.model, DIMS[args.model]
    out_path  = os.path.join(ENC_DIR, f'audio_{name}.npy')
    ckpt_path = os.path.join(ENC_DIR, f'audio_{name}_ckpt.npz')
    if os.path.exists(out_path):
        print(f'{name}: already complete'); return 0

    df = pd.read_parquet(os.path.join(OUT_DIR, 'merged_dataset.parquet'))
    audio_df = df[df['has_audio']].reset_index(drop=True)
    ids = pd.read_csv(os.path.join(ENC_DIR, 'audio_song_ids.csv'))['song_id'].tolist()
    assert audio_df['song_id'].tolist() == ids, 'row order changed'

    E = np.load(ckpt_path)['E'] if os.path.exists(ckpt_path) else \
        np.full((len(audio_df), 13, dim), np.nan, dtype=np.float32)
    todo = np.where(np.isnan(E[:, 0, 0]))[0]
    print(f'{name}: {len(audio_df) - len(todo)} done, {len(todo)} to go', flush=True)

    if len(todo):
        embed = load_model(name)
        batch = todo[:args.max_tracks]
        for j, i in enumerate(batch):
            y = librosa.load(audio_df.audio_path[i], sr=SR, mono=True, duration=DURATION)[0]
            wins, per_win = windows(y), []
            with torch.no_grad():
                for L in sorted({len(w) for w in wins}):        # equal-length windows only: no padding
                    hs = embed([w for w in wins if len(w) == L])
                    per_win += list(torch.stack([h.mean(1) for h in hs], 1).numpy())
            E[i] = np.mean(per_win, axis=0)
            if (j + 1) % CKPT_EVERY == 0 or j + 1 == len(batch):
                np.savez(ckpt_path, E=E)
                print(f'{name}: checkpoint, {int((~np.isnan(E[:, 0, 0])).sum())} done', flush=True)

    if np.isnan(E[:, 0, 0]).any():
        return 3
    assert not np.isnan(E).any()
    np.save(out_path, E)
    os.remove(ckpt_path)
    print(f'{name}: complete, saved {E.shape}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
