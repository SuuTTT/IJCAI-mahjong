#!/usr/bin/env node

const fs = require("fs");
const https = require("https");
const path = require("path");

const [inputPath, outputDir, targetPlayer, fileSlug] = process.argv.slice(2);
const tournament = "Simulation-8 / 模拟赛-8";
const tournamentId = "6a1d3d1258ebe27b197a8cca";

if (!inputPath || !outputDir || !targetPlayer || !fileSlug) {
  console.error(
    "Usage: node scripts/download_sim8_player.js <ids.json> <output_dir> <target_player> <file_slug>"
  );
  process.exit(1);
}

fs.mkdirSync(outputDir, { recursive: true });
const ids = JSON.parse(fs.readFileSync(inputPath, "utf8"));
const selected = [];
const failed = [];
let completed = 0;

function fetchJson(url, retries = 3) {
  return new Promise((resolve, reject) => {
    const request = https.get(url, { timeout: 30000 }, (response) => {
      if (response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
        response.resume();
        resolve(fetchJson(new URL(response.headers.location, url).href, retries));
        return;
      }

      let body = "";
      response.setEncoding("utf8");
      response.on("data", (chunk) => {
        body += chunk;
      });
      response.on("end", () => {
        try {
          resolve(JSON.parse(body));
        } catch (error) {
          reject(new Error(`Invalid JSON from ${url}: ${error.message}`));
        }
      });
    });

    request.on("timeout", () => request.destroy(new Error("Request timed out")));
    request.on("error", (error) => {
      if (retries > 1) {
        setTimeout(() => resolve(fetchJson(url, retries - 1)), 1000);
      } else {
        reject(error);
      }
    });
  });
}

async function main() {
  async function processMatch(matchId) {
    try {
      const litePath = path.join(outputDir, `${matchId}_lite.json`);
      const data = fs.existsSync(litePath)
        ? JSON.parse(fs.readFileSync(litePath, "utf8"))
        : await fetchJson(`https://www.botzone.org.cn/match/${matchId}?lite=true`);
      const playerNames = (data.players || []).map((player) => player.name);

      if (playerNames.includes(targetPlayer)) {
        fs.writeFileSync(litePath, JSON.stringify(data, null, 2));
        fs.writeFileSync(
          path.join(outputDir, `${matchId}_${fileSlug}_full_log.json`),
          JSON.stringify(data.logs, null, 2)
        );
        fs.writeFileSync(
          path.join(outputDir, `${matchId}_metadata.json`),
          JSON.stringify(
            {
              matchId,
              tournament,
              tournamentId,
              targetPlayer,
              status: data.status,
              success: data.success,
              players: data.players,
              initdata: data.initdata ? JSON.parse(data.initdata) : null,
              viewurl: data.viewurl,
              logRecords: Array.isArray(data.logs) ? data.logs.length : null,
              collectedAt: new Date().toISOString()
            },
            null,
            2
          )
        );
        selected.push(matchId);
      }
    } catch (error) {
      failed.push({ matchId, error: error.message });
      console.error(`${matchId} failed: ${error.message}`);
    } finally {
      completed++;
      console.log(`${completed}/${ids.length} ${matchId} selected=${selected.length}`);
    }
  }

  let nextIndex = 0;
  async function worker() {
    while (nextIndex < ids.length) {
      const index = nextIndex++;
      await processMatch(ids[index]);
    }
  }

  await Promise.all(Array.from({ length: 8 }, () => worker()));
  selected.sort();

  fs.writeFileSync(
    path.join(outputDir, "manifest.json"),
    JSON.stringify(
      {
        tournament,
        tournamentId,
        targetPlayer,
        fileSlug,
        scannedMatchCount: ids.length,
        selectedMatchCount: selected.length,
        selectedMatchIds: selected,
        failures: failed,
        collectedAt: new Date().toISOString()
      },
      null,
      2
    )
  );

  console.log(JSON.stringify({ scanned: ids.length, selected: selected.length, failed: failed.length }));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
