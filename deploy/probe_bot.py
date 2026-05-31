"""
probe_bot.py — throwaway Botzone bot to check the Python environment.

Upload as a Python bot and run one match. On its first turn it emits PASS, but
it writes a diagnostic to stderr (visible in Botzone's debug log / full log):
  PROBE numpy=<ver|MISSING> mahjong=<OK|MISSING> py=<version>

This tells you whether the real ML bot will have its dependencies on Botzone.
Always returns legal PASS so it never errors.
"""
import sys, json, platform

diag = {"py": platform.python_version()}
try:
    import numpy
    diag["numpy"] = numpy.__version__
except Exception as e:
    diag["numpy"] = "MISSING:%s" % e
try:
    import MahjongGB
    diag["mahjong"] = "OK"
except Exception as e:
    diag["mahjong"] = "MISSING:%s" % e

print("PROBE " + json.dumps(diag), file=sys.stderr)

# Always respond legally.
try:
    raw = sys.stdin.read()
    data = json.loads(raw)
    # first turn -> PASS is always legal
    print(json.dumps({"response": "PASS",
                      "debug": "PROBE " + json.dumps(diag)}))
except Exception:
    print(json.dumps({"response": "PASS"}))
