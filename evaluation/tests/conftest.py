import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "evaluation"
BACKEND_DIR = REPO_ROOT / "backend"

for path in (EVAL_DIR, BACKEND_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))