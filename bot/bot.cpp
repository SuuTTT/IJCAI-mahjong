/**
 * IJCAI Mahjong AI Bot - v0.1 (C++ Botzone submission)
 * Strategy: shanten-minimizing discard + safe HU guard (>= 8 fan)
 *
 * Build locally:
 *   g++ -O2 -std=c++14 -I/workspace/Chinese-Standard-Mahjong/fan-calculator-usage \
 *       -DLOCAL_BUILD bot.cpp -ljsoncpp -o bot
 *
 * Botzone: upload the amalgamated single-file version (bot_submit.cpp)
 */

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <cstring>
#include <cassert>
#include <climits>

#ifdef _BOTZONE_ONLINE
#include "jsoncpp/json.h"
#else
#include <jsoncpp/json/json.h>
#endif

// Include the official MahjongGB algorithm (fan + shanten)
#ifdef LOCAL_BUILD
#include "ChineseOfficialMahjongHelper/Classes/mahjong-algorithm/fan_calculator.h"
#include "ChineseOfficialMahjongHelper/Classes/mahjong-algorithm/fan_calculator.cpp"
#include "ChineseOfficialMahjongHelper/Classes/mahjong-algorithm/shanten.h"
#include "ChineseOfficialMahjongHelper/Classes/mahjong-algorithm/shanten.cpp"
#endif

using namespace std;

// ── Tile string <-> mahjong::tile_t mapping ──────────────────────────────────

static unordered_map<string, mahjong::tile_t> str2tile;

static void init_tile_map() {
    for (int i = 1; i <= 9; i++) {
        str2tile["W" + to_string(i)] = mahjong::make_tile(TILE_SUIT_CHARACTERS, i);
        str2tile["B" + to_string(i)] = mahjong::make_tile(TILE_SUIT_DOTS, i);
        str2tile["T" + to_string(i)] = mahjong::make_tile(TILE_SUIT_BAMBOO, i);
    }
    for (int i = 1; i <= 4; i++)
        str2tile["F" + to_string(i)] = mahjong::make_tile(TILE_SUIT_HONORS, i);
    for (int i = 1; i <= 3; i++)
        str2tile["J" + to_string(i)] = mahjong::make_tile(TILE_SUIT_HONORS, i + 4);
}

// ── Game state ────────────────────────────────────────────────────────────────

struct Pack {
    string type;   // "CHI" "PENG" "GANG"
    string tile;
    int offer;     // which player provided it (0-3)
};

static int myPlayerID = 0;
static int quan = 0;           // prevalent wind (0=East)
static vector<string> hand;    // concealed tiles
static vector<Pack> packs;     // declared packs
static int flowerCount = 0;
static string lastDiscard;
static int lastDiscardPid = -1;
static bool lastWasBugang = false;

// Tile counts for "shown" tiles (used for 绝张 check — not critical for v0.1)
static unordered_map<string, int> shownTile;

// ── Fan calculation ───────────────────────────────────────────────────────────

/**
 * Returns total fan count if the current hand can win with the given winTile.
 * hand must NOT include winTile (the fan calculator treats them separately).
 * Returns -1 if not a valid win or fan < 8.
 */
static int computeFan(const vector<string> &handTiles, const string &winTile,
                      bool isSelfDrawn, bool isAboutKong, bool isWallLast) {
    mahjong::calculate_param_t param;
    mahjong::fan_table_t fan_table;
    memset(&param, 0, sizeof(param));
    memset(&fan_table, 0, sizeof(fan_table));

    param.hand_tiles.tile_count = (int)handTiles.size();
    for (int i = 0; i < (int)handTiles.size(); i++) {
        auto it = str2tile.find(handTiles[i]);
        if (it == str2tile.end()) return -1;
        param.hand_tiles.standing_tiles[i] = it->second;
    }
    param.hand_tiles.pack_count = (int)packs.size();
    for (int i = 0; i < (int)packs.size(); i++) {
        const Pack &p = packs[i];
        mahjong::pack_t &dp = param.hand_tiles.fixed_packs[i];
        auto it = str2tile.find(p.tile);
        if (it == str2tile.end()) return -1;
        if (p.type == "PENG")
            dp = mahjong::make_pack((p.offer - myPlayerID + 4) % 4,
                                    PACK_TYPE_PUNG, it->second);
        else if (p.type == "GANG")
            dp = mahjong::make_pack((p.offer - myPlayerID + 4) % 4,
                                    PACK_TYPE_KONG, it->second);
        else // CHI
            dp = mahjong::make_pack(p.offer, PACK_TYPE_CHOW, it->second);
    }

    auto winIt = str2tile.find(winTile);
    if (winIt == str2tile.end()) return -1;
    param.win_tile = winIt->second;
    param.flower_count = flowerCount;

    if (isSelfDrawn)   param.win_flag |= WIN_FLAG_SELF_DRAWN;
    if (isWallLast)    param.win_flag |= WIN_FLAG_WALL_LAST;
    if (shownTile.count(winTile) && shownTile.at(winTile) == 3)
        param.win_flag |= WIN_FLAG_4TH_TILE;
    if (isAboutKong)   param.win_flag |= WIN_FLAG_ABOUT_KONG;

    param.prevalent_wind = (mahjong::wind_t)quan;
    param.seat_wind      = (mahjong::wind_t)myPlayerID;

    int re = mahjong::calculate_fan(&param, &fan_table);
    if (re < 8 + flowerCount) return -1;
    return re;
}

// ── Shanten calculation ───────────────────────────────────────────────────────

/**
 * Returns minimum shanten number for hand using regular form + 7-pairs.
 * -1 = already winning, 0 = tenpai.
 */
static int computeShanten(const vector<string> &tiles) {
    mahjong::hand_tiles_t ht;
    memset(&ht, 0, sizeof(ht));
    ht.pack_count = (int)packs.size();
    for (int i = 0; i < (int)packs.size(); i++) {
        const Pack &p = packs[i];
        auto it = str2tile.find(p.tile);
        if (it == str2tile.end()) return 8;
        if (p.type == "PENG")
            ht.fixed_packs[i] = mahjong::make_pack((p.offer - myPlayerID + 4) % 4,
                                                    PACK_TYPE_PUNG, it->second);
        else if (p.type == "GANG")
            ht.fixed_packs[i] = mahjong::make_pack((p.offer - myPlayerID + 4) % 4,
                                                    PACK_TYPE_KONG, it->second);
        else
            ht.fixed_packs[i] = mahjong::make_pack(p.offer, PACK_TYPE_CHOW, it->second);
    }
    ht.tile_count = (int)tiles.size();
    for (int i = 0; i < (int)tiles.size(); i++) {
        auto it = str2tile.find(tiles[i]);
        if (it == str2tile.end()) return 8;
        ht.standing_tiles[i] = it->second;
    }
    // Use basic_form_shanten and seven_pairs_shanten
    int s1 = mahjong::basic_form_shanten(ht.standing_tiles, ht.tile_count, nullptr);
    int s2 = mahjong::seven_pairs_shanten(ht.standing_tiles, ht.tile_count, nullptr);
    return min(s1, s2);
}

// ── Danger scoring ────────────────────────────────────────────────────────────
// Tracks how many of each tile have been publicly seen (discards + melds).
// A tile with shownTile[t]==3 is the last copy — very dangerous to discard.
// Returns 0-100 danger score for discarding tile t.
static int dangerScore(const string &t) {
    int seen = 0;
    auto it = shownTile.find(t);
    if (it != shownTile.end()) seen = it->second;
    // Last copy: extremely dangerous (probably someone is waiting for it)
    if (seen >= 3) return 80;
    // Honour tiles (wind/arrow): safe after 2 seen (only 4 copies total)
    if (t[0] == 'F' || t[0] == 'J') return seen * 10;
    // Number tiles: middle tiles (4-6) are more dangerous
    int n = t[1] - '0';
    int centre_penalty = max(0, 4 - abs(n - 5)) * 5;  // 0 for 1/9, 20 for 5
    return seen * 8 + centre_penalty;
}

/**
 * Score a candidate discard. Lower = better.
 * Combines shanten, expected fan at tenpai, and danger.
 */
static int discardScore(const vector<string> &remaining) {
    int s = computeShanten(remaining);

    // If tenpai (s == -1 means won, s == 0 means one tile away):
    // estimate expected fan by trying all 34 possible win tiles
    int fanBonus = 0;
    if (s == 0) {
        // Try each tile type as win tile; accumulate reachable fan
        vector<string> tileTypes;
        for (int i=1;i<=9;i++) { tileTypes.push_back("W"+to_string(i));
                                  tileTypes.push_back("B"+to_string(i));
                                  tileTypes.push_back("T"+to_string(i)); }
        for (int i=1;i<=4;i++) tileTypes.push_back("F"+to_string(i));
        for (int i=1;i<=3;i++) tileTypes.push_back("J"+to_string(i));

        int maxFan = 0, nWaits = 0;
        for (auto &wt : tileTypes) {
            int fan = computeFan(remaining, wt, false, false, false);
            if (fan >= 8) { maxFan = max(maxFan, fan); nWaits++; }
        }
        // Reward tenpai hands with high expected fan
        fanBonus = -(maxFan / 4 + nWaits);
    }

    return s * 200 + fanBonus;
}

/**
 * Pick best tile to discard: minimize (shanten, low fan, danger).
 */
static string bestDiscard() {
    if (hand.empty()) return "";
    int   bestScore = INT_MAX;
    int   bestDanger = INT_MAX;
    string bestTile = hand[0];

    for (int i = 0; i < (int)hand.size(); i++) {
        vector<string> tmp;
        for (int j = 0; j < (int)hand.size(); j++)
            if (j != i) tmp.push_back(hand[j]);
        int sc  = discardScore(tmp);
        int dng = dangerScore(hand[i]);
        int total = sc * 10 + dng;   // shanten/fan dominates; danger breaks ties
        if (total < bestScore || (total == bestScore && dng < bestDanger)) {
            bestScore  = total;
            bestDanger = dng;
            bestTile   = hand[i];
        }
    }
    return bestTile;
}

// ── Response builders ─────────────────────────────────────────────────────────

static string respondAfterDraw(const string &drawnTile) {
    // 1. Check HU (自摸): pass hand WITHOUT the drawn tile + drawnTile as winTile
    {
        vector<string> handWithoutWin;
        bool removed = false;
        for (auto &t : hand) {
            if (!removed && t == drawnTile) { removed = true; continue; }
            handWithoutWin.push_back(t);
        }
        int fan = computeFan(handWithoutWin, drawnTile, true, false, false);
        if (fan >= 8)
            return "HU";
    }

    // 2. Check BUGANG (upgrade a penged triplet to a kong)
    for (auto &p : packs) {
        if (p.type == "PENG" && p.tile == drawnTile) {
            // Check if bugang hurts shanten
            vector<string> testHand;
            for (auto &t : hand) if (t != drawnTile) testHand.push_back(t);
            // Remove only one copy
            bool removed = false;
            for (int i = 0; i < (int)testHand.size(); i++) {
                if (!removed && testHand[i] == drawnTile) {
                    testHand.erase(testHand.begin() + i);
                    removed = true;
                }
            }
            int sBefore = computeShanten(hand);
            int sAfter  = computeShanten(testHand);
            if (sAfter <= sBefore)
                return "BUGANG " + drawnTile;
        }
    }

    // 3. Check GANG (暗杠: have 4 copies of same tile)
    unordered_map<string, int> cnt;
    for (auto &t : hand) cnt[t]++;
    for (auto &kv : cnt) {
        if (kv.second >= 4) {
            // Check if angang doesn't break tenpai
            vector<string> testHand;
            for (auto &t : hand) if (t != kv.first) testHand.push_back(t);
            int sBefore = computeShanten(hand);
            int sAfter  = computeShanten(testHand);
            if (sAfter <= sBefore)
                return "GANG " + kv.first;
        }
    }

    // 4. Best discard
    return "PLAY " + bestDiscard();
}

static string respondAfterDiscard(int discardPid) {
    if (lastDiscard.empty()) return "PASS";
    const string &tile = lastDiscard;
    int nextPid = (discardPid + 1) % 4;

    // 1. Check HU (荣和): hand currently has 13 tiles (no draw), tile is win tile
    {
        int fan = computeFan(hand, tile, false, false, false);
        if (fan >= 8) return "HU";
    }

    // 2. Check GANG (have 3 in hand)
    int cnt3 = 0;
    for (auto &t : hand) if (t == tile) cnt3++;
    if (cnt3 >= 3) {
        vector<string> testHand;
        int removed = 0;
        for (auto &t : hand) {
            if (t == tile && removed < 3) { removed++; continue; }
            testHand.push_back(t);
        }
        int sBefore = computeShanten(hand);
        int sAfter  = computeShanten(testHand);
        if (sAfter <= sBefore)
            return "GANG";
    }

    // 3. Check PENG (have 2 in hand)
    int cnt2 = 0;
    for (auto &t : hand) if (t == tile) cnt2++;
    if (cnt2 >= 2) {
        vector<string> testHand;
        int removed = 0;
        for (auto &t : hand) {
            if (t == tile && removed < 2) { removed++; continue; }
            testHand.push_back(t);
        }
        int sBefore = computeShanten(hand);
        // Find best discard from testHand using full score (shanten + fan + danger)
        int bestScore = INT_MAX;
        string discardAfter;
        for (int i = 0; i < (int)testHand.size(); i++) {
            vector<string> tmp;
            for (int j = 0; j < (int)testHand.size(); j++)
                if (j != i) tmp.push_back(testHand[j]);
            int sc = discardScore(tmp) * 10 + dangerScore(testHand[i]);
            if (sc < bestScore) { bestScore = sc; discardAfter = testHand[i]; }
        }
        // Accept PENG if it improves shanten OR reaches tenpai with good fan
        int afterS = (discardAfter.empty()) ? 99 : computeShanten(
            [&]{ vector<string> t2; for(int j=0;j<(int)testHand.size();j++)
                     if(testHand[j]!=discardAfter) t2.push_back(testHand[j]); return t2; }()
        );
        if (afterS <= sBefore && !discardAfter.empty())
            return "PENG " + discardAfter;
    }

    // 4. Check CHI (only for the player immediately after discardPid)
    if (myPlayerID == nextPid && !tile.empty() &&
        (tile[0] == 'W' || tile[0] == 'B' || tile[0] == 'T'))
    {
        int n = tile[1] - '0';
        char suit = tile[0];

        // Try all possible middle tiles for the chi sequence
        // discard tile can be at position -1, 0, or +1 in sequence
        for (int midOff = -1; midOff <= 1; midOff++) {
            int midN = n + midOff;
            if (midN < 2 || midN > 8) continue;
            string midTile = string(1, suit) + to_string(midN);
            // Check if I have the other two tiles
            string needed[3];
            for (int d = -1; d <= 1; d++)
                needed[d + 1] = string(1, suit) + to_string(midN + d);
            // Count how many I need from hand (excluding discard tile)
            vector<string> handCopy = hand;
            bool canChi = true;
            for (auto &nt : needed) {
                if (nt == tile) continue; // provided by discard
                auto it = find(handCopy.begin(), handCopy.end(), nt);
                if (it == handCopy.end()) { canChi = false; break; }
                handCopy.erase(it);
            }
            if (!canChi) continue;

            // Simulate chi: handCopy is hand after chi, now find best discard
            int sBefore = computeShanten(hand);
            string discardAfter;
            int bestS = INT_MAX;
            for (int i = 0; i < (int)handCopy.size(); i++) {
                vector<string> tmp;
                for (int j = 0; j < (int)handCopy.size(); j++)
                    if (j != i) tmp.push_back(handCopy[j]);
                int s = computeShanten(tmp);
                if (s < bestS) { bestS = s; discardAfter = handCopy[i]; }
            }
            if (bestS < sBefore && !discardAfter.empty())
                return "CHI " + midTile + " " + discardAfter;
        }
    }

    return "PASS";
}

static string respondAfterGangNotify() {
    if (!lastWasBugang || lastDiscard.empty()) return "PASS";
    // 抢杠和 (rob the kong): hand has 13 tiles, lastDiscard is the bugang tile
    int fan = computeFan(hand, lastDiscard, false, /*aboutKong=*/true, false);
    if (fan >= 8)
        return "HU";
    return "PASS";
}

// ── Protocol parsing ──────────────────────────────────────────────────────────

// Apply previous request/response pair to update state
static void applyHistory(const string &req, const string &resp) {
    istringstream sin(req);
    int rtype;
    sin >> rtype;

    if (rtype == 1) {
        // Deal: "1 f0 f1 f2 f3 t1 t2 ... t13 [flowers]"
        int f0, f1, f2, f3;
        sin >> f0 >> f1 >> f2 >> f3;
        flowerCount = (myPlayerID == 0 ? f0 : myPlayerID == 1 ? f1 :
                       myPlayerID == 2 ? f2 : f3);
        hand.clear();
        for (int i = 0; i < 13; i++) {
            string t; sin >> t;
            hand.push_back(t);
        }
        return;
    }

    if (rtype == 2) {
        // My draw: "2 tile"
        string tile; sin >> tile;
        hand.push_back(tile);
        // Apply my response
        istringstream rsin(resp);
        string op; rsin >> op;
        if (op == "PLAY") {
            string t; rsin >> t;
            hand.erase(find(hand.begin(), hand.end(), t));
        } else if (op == "GANG") {
            string t; rsin >> t;
            // Remove 4 copies from hand
            for (int i = 0; i < 4; i++) {
                auto it = find(hand.begin(), hand.end(), t);
                if (it != hand.end()) hand.erase(it);
            }
            packs.push_back({"GANG", t, myPlayerID});
        } else if (op == "BUGANG") {
            string t; rsin >> t;
            auto it = find(hand.begin(), hand.end(), t);
            if (it != hand.end()) hand.erase(it);
            for (auto &p : packs)
                if (p.type == "PENG" && p.tile == t) { p.type = "GANG"; break; }
        }
        return;
    }

    if (rtype == 3) {
        // Notification: "3 pid ACTION [tile] [tile2]"
        int pid; string action, tile1, tile2;
        sin >> pid >> action;
        if (sin >> tile1) sin >> tile2;

        if (action == "PLAY") {
            lastDiscard = tile1;
            lastDiscardPid = pid;
            lastWasBugang = false;
            shownTile[tile1]++;
            if (pid == myPlayerID) return; // my own play, nothing to do
            // Apply my response
            istringstream rsin(resp);
            string op; rsin >> op;
            if (op == "PENG") {
                // Remove 2 copies from hand, add pack
                for (int i = 0; i < 2; i++) {
                    auto it = find(hand.begin(), hand.end(), tile1);
                    if (it != hand.end()) hand.erase(it);
                }
                packs.push_back({"PENG", tile1, pid});
                string discardT; rsin >> discardT;
                auto it = find(hand.begin(), hand.end(), discardT);
                if (it != hand.end()) hand.erase(it);
                shownTile[tile1] += 3;
            } else if (op == "GANG") {
                // Remove 3 copies from hand, add pack
                for (int i = 0; i < 3; i++) {
                    auto it = find(hand.begin(), hand.end(), tile1);
                    if (it != hand.end()) hand.erase(it);
                }
                packs.push_back({"GANG", tile1, pid});
                shownTile[tile1] = 4;
            } else if (op == "CHI") {
                string midTile, discardAfter; rsin >> midTile >> discardAfter;
                int midN = midTile[1] - '0';
                char suit = midTile[0];
                // Remove mid-1, mid, mid+1 from hand, except the discarded tile
                for (int d = -1; d <= 1; d++) {
                    string t = string(1, suit) + to_string(midN + d);
                    if (t == tile1) continue; // from discard
                    auto it = find(hand.begin(), hand.end(), t);
                    if (it != hand.end()) hand.erase(it);
                }
                // CHI offer: position of discard tile in sequence (1=left, 2=mid, 3=right)
                int offer = (tile1[1] - '0') - (midN - 1);
                packs.push_back({"CHI", midTile, offer});
                // Remove discard after chi
                auto it = find(hand.begin(), hand.end(), discardAfter);
                if (it != hand.end()) hand.erase(it);
            }
        } else if (action == "GANG") {
            // Another player declared concealed kong
            lastWasBugang = false;
        } else if (action == "BUGANG") {
            lastDiscard = tile1;
            lastDiscardPid = pid;
            lastWasBugang = (pid != myPlayerID);
        } else if (action == "PENG") {
            // Another player penged — update shown tiles
            if (!tile1.empty()) shownTile[tile1] += 3;
        } else if (action == "CHI") {
            // Another player chied
        }
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    init_tile_map();

    Json::Value inputJSON;
    cin >> inputJSON;

    int turnID = (int)inputJSON["responses"].size();

    vector<string> requests, responses;
    for (int i = 0; i < turnID; i++) {
        requests.push_back(inputJSON["requests"][i].asString());
        responses.push_back(inputJSON["responses"][i].asString());
    }
    requests.push_back(inputJSON["requests"][turnID].asString());

    // Request 0: init "0 playerID prevalentWind"
    if (!requests.empty()) {
        istringstream s0(requests[0]);
        int tmp; s0 >> tmp >> myPlayerID >> quan;
    }

    // Replay history
    for (int i = 1; i < turnID; i++) {
        applyHistory(requests[i], responses[i]);
    }

    // Current request
    string response = "PASS";
    if (turnID >= 1) {
        const string &curr = requests[turnID];
        istringstream sin(curr);
        int rtype; sin >> rtype;

        if (rtype == 1) {
            // Deal (shouldn't happen as current request — means history is short)
            response = "PASS";
        } else if (rtype == 2) {
            // My draw
            string tile; sin >> tile;
            hand.push_back(tile);
            response = respondAfterDraw(tile);
        } else if (rtype == 3) {
            int pid; string action, tile1;
            sin >> pid >> action;
            sin >> tile1;
            if (pid == myPlayerID) {
                response = "PASS";  // my own action notification
            } else if (action == "PLAY") {
                lastDiscard = tile1;
                lastDiscardPid = pid;
                lastWasBugang = false;
                response = respondAfterDiscard(pid);
            } else if (action == "BUGANG") {
                lastDiscard = tile1;
                lastDiscardPid = pid;
                lastWasBugang = true;
                response = respondAfterGangNotify();
            } else {
                response = "PASS";
            }
        }
    }

    Json::Value outputJSON;
    outputJSON["response"] = response;
    cout << outputJSON << endl;

    return 0;
}
