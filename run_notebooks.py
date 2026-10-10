"""Execute the analysis notebooks headlessly, in place, in paper order.

    python run_notebooks.py                                   # every notebook
    python run_notebooks.py phase5f_main_results phase6c_paradox_gallery
    python run_notebooks.py --from phase4b_coherence_rigor     # this one and everything after it

Notebooks run with the repository root as working directory, so relative paths resolve as in the paper.
"""
import argparse
import os
import sys
import time

import nbformat
from nbclient import NotebookClient

ROOT = os.path.dirname(os.path.abspath(__file__))
ORDER = [
    'phase1_data_alignment',
    'phase2_audio_features', 'phase2b_clap_deterministic', 'phase2c_mert_muq',
    'phase3_lyrics_features', 'phase3b_text_encoders',
    'phase4_similarity_engine', 'phase4b_coherence_rigor', 'phase4c_coherence_muq',
    'phase5_evaluation', 'phase5b_shared_split_stats', 'phase5c_graded_relevance', 'phase5d_clap_det_comparison',
    'phase5e_encoder_comparison', 'phase5f_main_results', 'phase5g_lyrics_chunking', 'phase5h_relevance_sensitivity',
    'phase6_visualization', 'phase6b_embedding_structure', 'phase6c_paradox_gallery',
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('notebooks', nargs='*', help='names without .ipynb (default: all)')
    ap.add_argument('--from', dest='start', help='run this notebook and every later one')
    ap.add_argument('--timeout', type=int, default=4 * 3600, help='per-cell timeout in seconds')
    args = ap.parse_args()

    names = [n.removesuffix('.ipynb') for n in args.notebooks] or ORDER
    if args.start:
        names = ORDER[ORDER.index(args.start.removesuffix('.ipynb')):]
    for name in names:
        if name not in ORDER:
            sys.exit(f'unknown notebook: {name}')

    os.environ.setdefault('MUSIC_RESEARCH_ROOT', ROOT)
    for name in names:
        path = os.path.join(ROOT, name + '.ipynb')
        nb = nbformat.read(path, as_version=4)
        t0 = time.time()
        print(f'{name} ...', flush=True)
        NotebookClient(nb, timeout=args.timeout, kernel_name='python3',
                       resources={'metadata': {'path': ROOT}}).execute()
        nbformat.write(nb, path)
        print(f'{name} done in {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
