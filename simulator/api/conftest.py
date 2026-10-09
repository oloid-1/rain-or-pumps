# lets `pytest simulator/api` run from the repo root: the tests import `app` directly
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
