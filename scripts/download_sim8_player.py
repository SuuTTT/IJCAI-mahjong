#!/usr/bin/env python3

import argparse
import concurrent.futures
import json
import time
import urllib.request
from pathlib import Path


TOURNAMENT = "Simulation-8 / 模拟赛-8"
TOURNAMENT_ID = "6a1d3d1258ebe27b197a8cca"


def fetch_json(url, retries=4):
    last_error = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "identity",
                    "User-Agent": "Mozilla/5.0 Botzone Sim8 player collector",
                },
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as error:
            last_error = error
            time.sleep(1 + attempt * 2)
    raise last_error


def write_json(path, data):
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def process_match(output_dir, target_player, file_slug, match_id):
    lite_path = output_dir / f"{match_id}_lite.json"
    data = (
        json.loads(lite_path.read_text(encoding="utf-8"))
        if lite_path.exists()
        else fetch_json(f"https://www.botzone.org.cn/match/{match_id}?lite=true")
    )
    if target_player not in [player.get("name") for player in data.get("players", [])]:
        return match_id, False

    write_json(lite_path, data)
    write_json(output_dir / f"{match_id}_{file_slug}_full_log.json", data.get("logs"))
    initdata = data.get("initdata")
    if isinstance(initdata, str):
        try:
            initdata = json.loads(initdata)
        except json.JSONDecodeError:
            pass
    write_json(
        output_dir / f"{match_id}_metadata.json",
        {
            "matchId": match_id,
            "tournament": TOURNAMENT,
            "tournamentId": TOURNAMENT_ID,
            "targetPlayer": target_player,
            "status": data.get("status"),
            "success": data.get("success"),
            "players": data.get("players"),
            "initdata": initdata,
            "viewurl": data.get("viewurl"),
            "logRecords": len(data["logs"]) if isinstance(data.get("logs"), list) else None,
            "collectedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    return match_id, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ids_json")
    parser.add_argument("output_dir")
    parser.add_argument("target_player")
    parser.add_argument("file_slug")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    match_ids = json.loads(Path(args.ids_json).read_text(encoding="utf-8"))
    selected = []
    failures = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_match, output_dir, args.target_player, args.file_slug, match_id
            ): match_id
            for match_id in match_ids
        }
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            match_id = futures[future]
            completed += 1
            try:
                _, included = future.result()
                if included:
                    selected.append(match_id)
                print(
                    f"{completed}/{len(match_ids)} {match_id} selected={len(selected)}",
                    flush=True,
                )
            except Exception as error:
                failures.append({"matchId": match_id, "error": str(error)})
                print(f"{completed}/{len(match_ids)} {match_id} failed: {error}", flush=True)

    selected.sort()
    write_json(
        output_dir / "manifest.json",
        {
            "tournament": TOURNAMENT,
            "tournamentId": TOURNAMENT_ID,
            "targetPlayer": args.target_player,
            "fileSlug": args.file_slug,
            "scannedMatchCount": len(match_ids),
            "selectedMatchCount": len(selected),
            "selectedMatchIds": selected,
            "failures": failures,
            "collectedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    print(
        json.dumps(
            {"scanned": len(match_ids), "selected": len(selected), "failed": len(failures)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
