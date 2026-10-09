"""
Builds everything the app serves: the training rows, the UI data and the forecast
table (about 20 minutes on a laptop CPU). The results in simulator/ui/data are
committed, so the app and Vercel run without this step.

    python scripts/build_app.py          (or: make app-data)
"""

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
env = {**os.environ, "DATA_DIR": os.environ.get("DATA_DIR", str(REPO / "data"))}

for step in (["models/build_sequences.py"],
             ["simulator/ui/build_ui_data.py", "--run", "simulator/artifacts"],
             ["simulator/forecast/build_forecast.py"]):
    print(f"\n$ DATA_DIR={env['DATA_DIR']} python {' '.join(step)}", flush=True)
    subprocess.run([sys.executable, *step], cwd=REPO, env=env, check=True)
