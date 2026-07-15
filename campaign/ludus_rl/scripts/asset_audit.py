"""CI gate: every asset file must have a manifest entry with license/provenance."""
import json
import sys
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "assets"
bad = []
for pack_dir in sorted(p for p in ASSETS.iterdir() if p.is_dir()):
    man = pack_dir / "manifest.json"
    if not man.exists():
        bad.append(f"{pack_dir.name}: missing manifest.json")
        continue
    entries = json.loads(man.read_text()).get("entries", {})
    for f in pack_dir.rglob("*"):
        if f.name == "manifest.json" or f.is_dir():
            continue
        rel = str(f.relative_to(pack_dir))
        e = entries.get(rel)
        if not e:
            bad.append(f"{pack_dir.name}/{rel}: no manifest entry")
        elif not (e.get("license") or e.get("provenance")):
            bad.append(f"{pack_dir.name}/{rel}: entry lacks license/provenance")
for b in bad:
    print("AUDIT FAIL:", b)
if bad:
    sys.exit(1)
print(f"asset audit OK")
