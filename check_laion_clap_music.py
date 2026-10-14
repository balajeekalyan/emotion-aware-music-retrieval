"""Check the ORIGINAL LAION CLAP music checkpoint against its Hugging Face conversion (laion/larger_clap_music).

Phase 2d/2e found the HF conversion collapsed (unrelated text prompts at cosine 0.999, zero-shot genre at chance).
This script loads the original checkpoint `music_audioset_epoch_15_esc_90.14.pt` (huggingface.co/lukewys/laion_clap,
the LAION repository's music checkpoint; HTSAT-base) with the authors' `laion_clap` package and runs the phase2e
zero-shot check on the same 339 training tracks (`outputs/clap_checkpoint_check_sample.csv`, first 10 s), plus a
direct comparison of selected tensors with the HF conversion. The HF model card does not name its source checkpoint.

laion_clap needs its own environment (it conflicts with the main one), e.g.:
    python -m venv clapenv
    clapenv/Scripts/python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
    clapenv/Scripts/python -m pip install laion_clap==1.1.7 transformers==4.44.2 "numpy<2" librosa pandas pyarrow huggingface_hub
    clapenv/Scripts/python check_laion_clap_music.py <dir containing laion_ckpt/music_audioset_epoch_15_esc_90.14.pt>

The checkpoint is a pickle; it is loaded with an unpickler restricted to the globals it was found to contain.
Writes outputs/clap_laion_original_music_check.csv and outputs/clap_laion_original_vs_hf_weights.json.
"""
import glob
import json
import os
import sys

import librosa
import numpy as np
import pandas as pd
import torch
import laion_clap
from safetensors.torch import load_file

S = sys.argv[1]
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, 'outputs')
CKPT = os.path.join(S, 'laion_ckpt', 'music_audioset_epoch_15_esc_90.14.pt')
HF_DIR = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--laion--larger_clap_music/snapshots/*/model.safetensors'))[0]
QUADS = ['Q1', 'Q2', 'Q3', 'Q4']
GENRES = ['Pop/Rock', 'Electronic', 'Country', 'Rap', 'Blues', 'Jazz', 'Reggae']
PROMPTS = {
    'Q1': ['happy music', 'upbeat, joyful music', 'cheerful and energetic music', 'exciting, euphoric music'],
    'Q2': ['angry music', 'aggressive, tense music', 'furious and intense music', 'anxious, menacing music'],
    'Q3': ['sad music', 'melancholic, depressing music', 'sorrowful and gloomy music', 'lonely, heartbroken music'],
    'Q4': ['calm music', 'relaxing, peaceful music', 'serene and soothing music', 'tender, content music'],
}
UNRELATED = ['a dog barking', 'heavy metal guitar solo', 'a quiet string quartet', 'rain on a window',
             'a man giving a speech', 'electronic dance music with a strong beat']
l2 = lambda X: X / np.linalg.norm(X, axis=-1, keepdims=True)
torch.set_num_threads(os.cpu_count())

import pickle
import types
ALLOWED = {('collections', 'OrderedDict'), ('numpy', 'dtype'), ('numpy.core.multiarray', 'scalar'), ('_codecs', 'encode'),
           ('torch', 'FloatStorage'), ('torch', 'LongStorage'), ('torch._utils', '_rebuild_tensor_v2')}


class RestrictedUnpickler(pickle.Unpickler):
    """Only the globals found by static inspection of this checkpoint's data.pkl; anything else is refused."""
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f'refused global {module}.{name}')
        return super().find_class(module, name)


restricted = types.ModuleType('restricted_pickle')
restricted.Unpickler = RestrictedUnpickler
restricted.load = pickle.load
ck = torch.load(CKPT, map_location='cpu', weights_only=False, pickle_module=restricted)
print('loaded with restricted unpickler; top-level keys:', list(ck)[:10])
sd = ck['state_dict'] if 'state_dict' in ck else ck
sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}

model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
res = model.model.load_state_dict(sd, strict=False)
print('missing keys:', res.missing_keys, '| unexpected keys:', res.unexpected_keys)
model.eval()

# ---- weights: original vs HF conversion ----
hf = load_file(HF_DIR)
cmp = {'logit_scale_a (exp)': (float(sd['logit_scale_a'].exp()), float(hf['logit_scale_a'].exp())),
       'logit_scale_t (exp)': (float(sd['logit_scale_t'].exp()), float(hf['logit_scale_t'].exp()))}
pairs = [('text_branch.embeddings.word_embeddings.weight', 'text_model.embeddings.word_embeddings.weight'),
         ('text_projection.0.weight', 'text_projection.linear1.weight'),
         ('text_projection.2.weight', 'text_projection.linear2.weight'),
         ('audio_projection.0.weight', 'audio_projection.linear1.weight'),
         ('audio_projection.2.weight', 'audio_projection.linear2.weight')]
for a, b in pairs:
    if a in sd and b in hf:
        x, y = sd[a].float(), hf[b].float()
        cmp[a] = ('max |orig - hf|', float((x - y).abs().max()) if x.shape == y.shape else f'shape {tuple(x.shape)} vs {tuple(y.shape)}')
    else:
        cmp[a] = ('key missing', a in sd, b in hf)
print(json.dumps(cmp, indent=1))

# ---- zero-shot check on the phase2e training-track sample ----
smp = pd.read_csv(os.path.join(OUT_DIR, 'clap_checkpoint_check_sample.csv'))
paths = pd.read_parquet(os.path.join(OUT_DIR, 'merged_dataset.parquet'), columns=['song_id', 'audio_path']).set_index('song_id')
smp['audio_path'] = paths.loc[smp.song_id, 'audio_path'].values
ids = smp.drop_duplicates('song_id')
wave = {r.song_id: librosa.load(r.audio_path, sr=48000, mono=True, duration=10)[0] for r in ids.itertuples()}
q = lambda x: (np.clip(x, -1, 1) * 32767).astype(np.int16).astype(np.float32) / 32767   # README's int16 quantization
A = {}
order = list(wave)
for i in range(0, len(order), 8):
    b = order[i:i + 8]
    L = 480000
    X = np.stack([np.pad(q(wave[s])[:L], (0, max(0, L - len(wave[s][:L])))) for s in b])
    with torch.no_grad():
        E = model.get_audio_embedding_from_data(x=X, use_tensor=False)
    A.update(zip(b, l2(E)))
def text(t):
    with torch.no_grad():
        return l2(model.get_text_embedding(t, use_tensor=False))
T = text(UNRELATED); tc = (T @ T.T)[np.triu_indices(len(T), 1)].mean()
Aall = np.vstack([A[s] for s in order]); ac = np.median((Aall @ Aall.T)[np.triu_indices(len(Aall), 1)])
g = smp[smp.task == 'genre']; m = smp[smp.task == 'mood']
G = text([f'{x} music' for x in GENRES])
gp = np.array(GENRES)[(np.vstack([A[s] for s in g.song_id]) @ G.T).argmax(1)]
Qm = l2(np.vstack([text(PROMPTS[x]).mean(0) for x in QUADS]))
qp = np.array(QUADS)[(np.vstack([A[s] for s in m.song_id]) @ Qm.T).argmax(1)]
row = {'checkpoint': 'laion_clap original: music_audioset_epoch_15_esc_90.14.pt (HTSAT-base)',
       'logit_scale_a': float(sd['logit_scale_a'].exp()), 'text_unrelated_cos_mean': float(tc),
       'audio_pair_cos_median': float(ac), 'genre_zero_shot_acc': float(np.mean(gp == g.label.values)),
       'genre_chance': 1 / len(GENRES), 'genre_predicted_classes': len(set(gp)),
       'mood_zero_shot_acc': float(np.mean(qp == m.label.values)),
       **{f'mood_pred_share_{x}': float(np.mean(qp == x)) for x in QUADS}}
print(json.dumps(row, indent=1))
pd.DataFrame([row]).to_csv(os.path.join(OUT_DIR, 'clap_laion_original_music_check.csv'), index=False)
json.dump(cmp, open(os.path.join(OUT_DIR, 'clap_laion_original_vs_hf_weights.json'), 'w'), indent=1)
