#!/usr/bin/env python3

import argparse
import concurrent.futures
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path


BASE_URL = "https://www.botzone.org.cn"
GAME_ID = "5e37dcf74019f43051e53201"
START_URL = f"{BASE_URL}/globalmatchlist?game={GAME_ID}"
TARGETS = {
    "[mythos]mythos",
    "[aidenh]hhhhhhhhh",
    "[Infunus]TypeC青雀",
}
TARGET_KEYS = [re.fullmatch(r"\[(.+)](.+)", target).groups() for target in TARGETS]
# Oldest creation ID among the three exact bots. No matching game can predate it.
OLDEST_TARGET_BOT_ID = "6a26bbce5eab685a5f7fefd8"


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_json(path, data, compact=False):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as output:
        json.dump(
            data,
            output,
            ensure_ascii=False,
            indent=None if compact else 2,
            separators=(",", ":") if compact else None,
        )
    temp.replace(path)


def fetch(url, as_json=False, retries=6):
    last_error = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json" if as_json else "text/html",
                    "Accept-Encoding": "identity",
                    "User-Agent": "Mozilla/5.0 Botzone target public-game collector",
                },
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                body = response.read().decode("utf-8")
                return json.loads(body) if as_json else body
        except Exception as error:
            last_error = error
            time.sleep(min(15, 1 + attempt * 2))
    raise last_error


class MatchListParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.matches = []
        self.next_url = None
        self.row = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = {"matchId": None, "text": []}
            return
        if tag != "a":
            return
        href = dict(attrs).get("href", "")
        match = re.fullmatch(r"/match/([0-9a-f]{24})", href)
        if match and self.row is not None:
            self.row["matchId"] = match.group(1)
        if href.startswith("/globalmatchlist?startid=") and f"game={GAME_ID}" in href:
            self.next_url = urllib.parse.urljoin(BASE_URL, href)

    def handle_data(self, data):
        if self.row is not None:
            self.row["text"].append(data)

    def handle_endtag(self, tag):
        if tag == "tr" and self.row is not None:
            if self.row["matchId"]:
                self.row["text"] = " ".join("".join(self.row["text"]).split())
                self.matches.append(self.row)
            self.row = None


def process_match(root, match_id):
    data = fetch(f"{BASE_URL}/match/{match_id}?lite=true", as_json=True)
    player_names = {player.get("name") for player in data.get("players", [])}
    matched = sorted(TARGETS & player_names)
    if not matched:
        return None

    log_path = root / "logs" / f"{match_id}_full_log.json"
    metadata_path = root / "metadata" / f"{match_id}_metadata.json"
    atomic_json(log_path, data.get("logs"), compact=True)
    initdata = data.get("initdata")
    if isinstance(initdata, str):
        try:
            initdata = json.loads(initdata)
        except json.JSONDecodeError:
            pass
    atomic_json(
        metadata_path,
        {
            "matchId": match_id,
            "matchedTargets": matched,
            "players": data.get("players"),
            "status": data.get("status"),
            "success": data.get("success"),
            "initdata": initdata,
            "viewurl": data.get("viewurl"),
            "logRecords": len(data["logs"]) if isinstance(data.get("logs"), list) else None,
            "collectedAt": utc_now(),
        },
        compact=True,
    )
    return matched


def initialize(root):
    status_path = root / "status.json"
    if status_path.exists():
        return json.loads(status_path.read_text(encoding="utf-8"))
    return {
        "version": 1,
        "criteria": "All public Chinese-Standard-Mahjong matches containing at least one exact target display name.",
        "targets": sorted(TARGETS),
        "startedAt": utc_now(),
        "pagesScanned": 0,
        "matchesScanned": 0,
        "candidateMatches": 0,
        "selectedMatches": 0,
        "targetAppearances": {target: 0 for target in sorted(TARGETS)},
        "failures": [],
        "nextUrl": START_URL,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    root = Path(args.output_root)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    status_path = root / "status.json"
    status = initialize(root)
    url = status.get("nextUrl") or START_URL

    while url:
        try:
            page = MatchListParser()
            page.feed(fetch(url))
            if not page.matches:
                status["stopReason"] = "no_matches_on_page"
                break

            in_range = [
                match
                for match in page.matches
                if match["matchId"] >= OLDEST_TARGET_BOT_ID
            ]
            candidates = [
                match["matchId"]
                for match in in_range
                if any(
                    author in match["text"] and bot_name in match["text"]
                    for author, bot_name in TARGET_KEYS
                )
            ]
            status["candidateMatches"] += len(candidates)
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(process_match, root, match_id): match_id
                    for match_id in candidates
                }
                for future in concurrent.futures.as_completed(futures):
                    match_id = futures[future]
                    try:
                        matched = future.result()
                        if matched:
                            status["selectedMatches"] += 1
                            for target in matched:
                                status["targetAppearances"][target] += 1
                            print(f"selected {match_id} {' '.join(matched)}", flush=True)
                    except Exception as error:
                        status["failures"].append(
                            {"matchId": match_id, "error": str(error)}
                        )

            status["pagesScanned"] += 1
            status["matchesScanned"] += len(in_range)
            status["nextUrl"] = page.next_url
            status["updatedAt"] = utc_now()
            atomic_json(status_path, status)
            print(
                f"pages={status['pagesScanned']} scanned={status['matchesScanned']} "
                f"candidates={status['candidateMatches']} selected={status['selectedMatches']} "
                f"failures={len(status['failures'])}",
                flush=True,
            )
            if len(in_range) < len(page.matches):
                status["stopReason"] = "reached_oldest_target_bot_creation"
                break
            url = page.next_url
        except Exception as error:
            status["failures"].append({"url": url, "error": str(error)})
            status["updatedAt"] = utc_now()
            atomic_json(status_path, status)
            print(f"page failed {url}: {error}", flush=True)
            time.sleep(20)
    else:
        status["stopReason"] = "no_older_page"

    status["nextUrl"] = url
    status["finishedAt"] = utc_now()
    atomic_json(status_path, status)


if __name__ == "__main__":
    main()
