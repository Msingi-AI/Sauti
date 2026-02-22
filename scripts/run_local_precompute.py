import os
import sys
from pathlib import Path

# Ensure local `src/` is importable when running from repo root
repo_root = Path(__file__).resolve().parent.parent
src_path = str(repo_root / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Force fake activations for a fast local smoke test
os.environ['USE_FAKE_ACTIVATIONS'] = '1'

from sauti.precompute import precompute_teacher_activations

if __name__ == '__main__':
    out_dir = os.path.join(os.getcwd(), 'checkpoints', 'teacher_activations')
    print('Running local precompute smoke test ->', out_dir)
    os.makedirs(out_dir, exist_ok=True)
    precompute_teacher_activations('google/WaxalNLP', 'swa_tts', out_dir, max_items=5)
    print('Done.')
