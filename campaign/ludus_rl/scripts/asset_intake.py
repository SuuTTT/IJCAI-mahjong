"""Register an asset file into its pack manifest.

python scripts/asset_intake.py boom-base ground_grass.png --tool gpt4o \
    --prompt-id T1 [--license CC0 --source URL]
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "assets"
ap = argparse.ArgumentParser()
ap.add_argument("pack")
ap.add_argument("file")
ap.add_argument("--tool", default=None)
ap.add_argument("--prompt-id", default=None)
ap.add_argument("--license", default=None)
ap.add_argument("--source", default=None)
a = ap.parse_args()
man_p = ASSETS / a.pack / "manifest.json"
man = json.loads(man_p.read_text())
f = ASSETS / a.pack / a.file
entry = {"sha256": hashlib.sha256(f.read_bytes()).hexdigest()[:16],
         "added": time.strftime("%Y-%m-%d")}
if a.tool:
    entry["provenance"] = {"tool": a.tool, "prompt_id": a.prompt_id}
if a.license:
    entry["license"] = a.license
    entry["source"] = a.source
man["entries"][a.file] = entry
man_p.write_text(json.dumps(man, indent=1))
print("registered", a.pack + "/" + a.file)
