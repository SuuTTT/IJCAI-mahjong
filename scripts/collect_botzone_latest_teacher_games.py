#!/usr/bin/env python3

import argparse
import concurrent.futures
import gzip
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
SIM8_CONTEST_ID = "6a1d3d1258ebe27b197a8cca"
TARGETS = {
    ("ppt", "Stella_R_2"),
    ("yxcatqwq", "学习困难麻将猫"),
    ("SYlastime2", "Stella_R_1"),
    ("Lastime11", "Stella_R_3"),
}


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def object_id_time(value):
    return datetime.fromtimestamp(int(value[:8], 16), timezone.utc)


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
                    "User-Agent": "Mozilla/5.0 Botzone latest-teacher collector",
                },
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                body = response.read()
                if body.startswith(b"\x1f\x8b"):
                    body = gzip.decompress(body)
                body = body.decode("utf-8")
                return json.loads(body) if as_json else body
        except Exception as error:
            last_error = error
            time.sleep(min(15, 1 + attempt * 2))
    raise last_error


class RankParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "tr" and attrs.get("data-botid"):
            self.row = {"botId": attrs["data-botid"], "cells": [], "versionId": None}
        elif tag == "td" and self.row is not None:
            self.cell = []
        elif tag == "button" and self.row is not None:
            onclick = attrs.get("onclick", "")
            match = re.search(r"Botzone\.copy\('([0-9a-f]{24})'", onclick)
            if match:
                self.row["versionId"] = match.group(1)

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag == "td" and self.cell is not None:
            self.row["cells"].append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


class MatchListParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.match_ids = []
        self.next_url = None

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        href = dict(attrs).get("href", "")
        match = re.fullmatch(r"/match/([0-9a-f]{24})", href)
        if match and match.group(1) not in self.match_ids:
            self.match_ids.append(match.group(1))
        if href.startswith("/globalmatchlist?startid=") and f"game={GAME_ID}" in href:
            self.next_url = urllib.parse.urljoin(BASE_URL, href)


def discover_latest_versions():
    detail = fetch(f"{BASE_URL}/contest/detail/{SIM8_CONTEST_ID}", as_json=True)
    found = {}
    for player in detail["contest"]["players"]:
        submitted = player["bot"]
        key = (submitted["user"]["name"], submitted["bot"]["name"])
        if key not in TARGETS:
            continue
        uploaded = object_id_time(submitted["_id"])
        found[key] = {
            "author": key[0],
            "botName": key[1],
            "displayName": f"[{key[0]}]{key[1]}",
            "botId": submitted["bot"]["_id"],
            "latestVersion": submitted["ver"],
            "latestVersionId": submitted["_id"],
            "latestVersionUploadedAt": uploaded.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "latestVersionUploadedTimestamp": int(uploaded.timestamp()),
            "versionSource": f"Simulation-8 submission {SIM8_CONTEST_ID}",
            "ranked": submitted["bot"].get("ranked"),
        }
    missing = TARGETS - set(found)
    if missing:
        raise RuntimeError(f"Could not find latest versions for: {sorted(missing)}")
    return found


def process_match(root, match_id, versions):
    match_time = object_id_time(match_id)
    data = fetch(f"{BASE_URL}/match/{match_id}?lite=true", as_json=True)
    player_names = {player.get("name") for player in data.get("players", [])}
    matched = [
        version
        for version in versions.values()
        if version["displayName"] in player_names
        and match_time.timestamp() >= version["latestVersionUploadedTimestamp"]
    ]
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
            "matchCreatedAt": match_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
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
    return [item["displayName"] for item in matched]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    root = Path(args.output_root)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    status_path = root / "status.json"

    versions = discover_latest_versions()
    atomic_json(root / "latest_versions.json", list(versions.values()))
    oldest_upload = min(
        item["latestVersionUploadedTimestamp"] for item in versions.values()
    )
    counts = {item["displayName"]: 0 for item in versions.values()}
    status = {
        "version": 1,
        "criteria": "Match contains at least one target after that target's latest-version upload time.",
        "latestVersions": list(versions.values()),
        "startedAt": utc_now(),
        "pagesScanned": 0,
        "matchesScanned": 0,
        "selectedMatches": 0,
        "targetAppearances": counts,
        "failures": [],
        "nextUrl": START_URL,
    }
    atomic_json(status_path, status)

    url = START_URL
    while url:
        page = MatchListParser()
        page.feed(fetch(url))
        if not page.match_ids:
            status["stopReason"] = "no_matches_on_page"
            break

        eligible_ids = [
            match_id
            for match_id in page.match_ids
            if object_id_time(match_id).timestamp() >= oldest_upload
        ]
        if eligible_ids:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(process_match, root, match_id, versions): match_id
                    for match_id in eligible_ids
                }
                for future in concurrent.futures.as_completed(futures):
                    match_id = futures[future]
                    try:
                        matched = future.result()
                        if matched:
                            status["selectedMatches"] += 1
                            for name in matched:
                                counts[name] += 1
                            print(f"selected {match_id} {' '.join(matched)}", flush=True)
                    except Exception as error:
                        status["failures"].append(
                            {"matchId": match_id, "error": str(error)}
                        )

        status["pagesScanned"] += 1
        status["matchesScanned"] += len(eligible_ids)
        status["targetAppearances"] = counts
        status["nextUrl"] = page.next_url
        status["updatedAt"] = utc_now()
        atomic_json(status_path, status)
        print(
            f"pages={status['pagesScanned']} scanned={status['matchesScanned']} "
            f"selected={status['selectedMatches']} failures={len(status['failures'])}",
            flush=True,
        )
        if len(eligible_ids) < len(page.match_ids):
            status["stopReason"] = "reached_oldest_latest_version_upload"
            break
        url = page.next_url
    else:
        status["stopReason"] = "no_older_page"

    status["nextUrl"] = url
    status["finishedAt"] = utc_now()
    atomic_json(status_path, status)


if __name__ == "__main__":
    main()
