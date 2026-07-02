"""provenance.py — embed file hashes + script hash into gate result dicts.
Usage: out.update(provenance(__file__, model_paths=[...]))  (best-practice fix, 2026-07-02)"""
import hashlib, os
def _sha(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(n)
            if not b: break
            h.update(b)
    return h.hexdigest()[:16]
def provenance(script, model_paths=()):
    return {"script_sha": _sha(script),
            "model_shas": {os.path.basename(p): _sha(p) for p in model_paths if os.path.exists(p)}}
