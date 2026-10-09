"""
Builds everything the app serves, from the bundled sample (data/sample/): the
training rows, the UI data and the forecast table. Vercel runs this as the build
step (vercel.json); locally it is `python scripts/build_app.py` or `make sample-app`.

    DATA_DIR=data python scripts/build_app.py       the same on the full data
"""

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
env = {**os.environ, "DATA_DIR": os.environ.get("DATA_DIR", str(REPO / "data" / "sample"))}

for step in (["models/build_sequences.py"],
             ["simulator/ui/build_ui_data.py", "--run", "simulator/artifacts"],
             ["simulator/forecast/build_forecast.py"]):
    print(f"\n$ DATA_DIR={env['DATA_DIR']} python {' '.join(step)}", flush=True)
    subprocess.run([sys.executable, *step], cwd=REPO, env=env, check=True)
