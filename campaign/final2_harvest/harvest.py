#!/usr/bin/env python3
"""Resumable polite harvester for Final2 match logs (READ-ONLY downloads)."""
import json, gzip, os, sys, time, threading, queue
import urllib.request

BASE = "/root/final2_harvest"
SID = json.load(open("/root/IJCAI-mahjong-full/realfield_build/moyu_sid.json"))["sid"]
HDRS = {
    "Cookie": f"connect.sid={SID}; locale=cn",
    "Accept-Encoding": "gzip",
    "User-Agent": "Mozilla/5.0 (harvest; contact moyu)",
}
CONC = 5
BATCH = 1024

# load mids in table order
mids = []
with gzip.open(f"{BASE}/table_rows.jsonl.gz", "rt") as f:
    for line in f:
        mids.append(json.loads(line)["mid"])
assert len(mids) == 12288, len(mids)

done = set()
ckpt = f"{BASE}/done_mids.txt"
if os.path.exists(ckpt):
    done = set(open(ckpt).read().split())
print(f"{len(done)} already done", flush=True)

os.makedirs(f"{BASE}/raw", exist_ok=True)

todo = [m for m in mids if m not in done]
q = queue.Queue()
for m in todo:
    q.put(m)

lock = threading.Lock()
results = {}   # mid -> dict
errors = {}    # mid -> errstr
ckpt_f = open(ckpt, "a")

def fetch(mid):
    url = f"https://www.botzone.org.cn/match/{mid}?lite=true"
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=HDRS)
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    data = gzip.decompress(data)
                d = json.loads(data)
                if not d.get("success", True) and "logs" not in d:
                    raise RuntimeError("api not success")
                return d
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))

def worker():
    while True:
        try:
            mid = q.get_nowait()
        except queue.Empty:
            return
        try:
            d = fetch(mid)
            d["_mid"] = mid
            with lock:
                results[mid] = d
        except Exception as e:
            with lock:
                errors[mid] = str(e)
        time.sleep(0.15)

def flush_batch():
    """Write accumulated results to next batch file, in table order."""
    global results
    with lock:
        ready = dict(results)
        results = {}
    if not ready:
        return 0
    # append to a rolling batch file numbered by count of existing
    existing = sorted(os.listdir(f"{BASE}/raw"))
    n = len(existing)
    path = f"{BASE}/raw/batch_{n:04d}.jsonl.gz"
    with gzip.open(path, "wt") as f:
        for mid, d in ready.items():
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    for mid in ready:
        ckpt_f.write(mid + "\n")
    ckpt_f.flush()
    return len(ready)

start = time.time()
total_done = 0
while not q.empty() or any(t.is_alive() for t in threading.enumerate() if t.name.startswith("hw")):
    threads = [threading.Thread(target=worker, name=f"hw{i}", daemon=True) for i in range(CONC)]
    for t in threads:
        t.start()
    # flush every ~BATCH results
    while any(t.is_alive() for t in threads):
        time.sleep(5)
        with lock:
            nready = len(results)
        if nready >= BATCH:
            n = flush_batch()
            total_done += n
            el = time.time() - start
            print(f"[{el:.0f}s] flushed {n}, total {total_done}/{len(todo)}, errors {len(errors)}", flush=True)
    n = flush_batch()
    total_done += n
    break

print(f"DONE fetched={total_done} errors={len(errors)} elapsed={time.time()-start:.0f}s", flush=True)
if errors:
    json.dump(errors, open(f"{BASE}/fetch_errors.json", "w"), indent=1)
    if len(errors) > 0.05 * len(mids):
        print("LOUD FAIL: >5% fetch errors", flush=True)
        sys.exit(2)
