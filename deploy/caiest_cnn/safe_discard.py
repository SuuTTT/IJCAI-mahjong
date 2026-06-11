"""
safe_discard.py — defensive discard filter (Tier-2 #F1), deploy-time, no training.
Grounded in a HARD rule, not a fuzzy heuristic: a tile in opponent p's discard pile can never win
them by rong (furiten) -> it is 100% SAFE against p (genbutsu). When a committed opponent is likely
threatening, prefer the policy's highest-ranked GENBUTSU discard (within its top-K), trading a little
offense for not dealing in. If no threat, or no safe tile in top-K, keep the policy's choice.

Threat model (v1, conservative): an opponent is "threatening" if they have melded (revealed pung/chi/
gang => committed to a hand) AND the game is past the opening (>= MIN_TURN own discards). Crude but
safe-leaning; the real validation is the ladder (deal-in rate). OFF by default (SAFE_DISCARD=1).

agent = FeatureAgent (feature.py): agent.history[p] = player p's discards (p relative, 0=self),
agent.packs[p] = player p's melds.
"""
import os
MIN_TURN = int(os.environ.get("SAFE_MIN_TURN", "12"))  # only defend in the LATE game
TOP_K = int(os.environ.get("SAFE_TOP_K", "2"))         # sacrifice at most a tiny bit of offense
MIN_MELDS = int(os.environ.get("SAFE_MIN_MELDS", "2")) # only vs STRONGLY committed opponents (2+ melds)

# FOLD mode (SAFE_DISCARD=2): when our hand is a DEAD SHAPE (too far from a winning hand late in the
# game) its offensive value is ~zero — so stop feeding: pick the SAFEST discard from the FULL legal
# list (genbutsu vs threats, else most-visible tile), not just a top-K swap. "Target second place":
# in duplicate scoring the only controllable extra loss is being the rong feeder (pays 8+fan more).
FOLD_TURN = int(os.environ.get("SAFE_FOLD_TURN", "9"))       # don't fold before this many own discards
FOLD_SHANTEN = int(os.environ.get("SAFE_FOLD_SHANTEN", "3")) # fold if best shanten >= this, late game
try:
    from MahjongGB import MahjongShanten
except Exception:
    MahjongShanten = None


def _best_shanten(agent):
    """Min shanten over all discards from the current (just-drew) hand. 99 if unknown."""
    if MahjongShanten is None:
        return 99
    pk = tuple(agent.packs[0])
    best = 99
    for t in set(agent.hand):
        rem = list(agent.hand); rem.remove(t)
        try:
            s = MahjongShanten(pack=pk, hand=tuple(rem))
        except Exception:
            continue
        if s < best:
            best = s
    return best


def _fold_discard(agent, ranked_tiles):
    """Full-defense discard: highest-ranked tile that is genbutsu vs every melded opponent; else the
    most-visible tile (fewest live copies -> fewest waits it can complete)."""
    hist = agent.history
    threats = [p for p in (1, 2, 3) if len(agent.packs[p]) >= 1] or [1, 2, 3]
    for t in ranked_tiles:
        if all(t in hist[p] for p in threats):
            return t
    shown = agent.shownTiles
    return max(ranked_tiles, key=lambda t: (shown.get(t, 0), -ranked_tiles.index(t)))


def choose_discard(agent, ranked_tiles, top_k=TOP_K):
    """ranked_tiles: legal discard tiles, policy order (best first). Return the discard to play."""
    hist = agent.history; packs = agent.packs
    if os.environ.get("SAFE_DISCARD") == "2" and len(hist[0]) >= FOLD_TURN:
        if _best_shanten(agent) >= FOLD_SHANTEN:        # dead shape late -> full defense
            return _fold_discard(agent, ranked_tiles)
    if len(hist[0]) < MIN_TURN:                         # opening: play the policy's choice
        return ranked_tiles[0]
    threats = [p for p in (1, 2, 3) if len(packs[p]) >= MIN_MELDS]   # strongly-committed opponents
    if not threats:
        return ranked_tiles[0]
    def genbutsu(t):                                    # safe vs ALL threats (in each one's discards)
        return all(t in hist[p] for p in threats)
    for t in ranked_tiles[:top_k]:
        if genbutsu(t):
            return t                                    # highest-ranked safe tile
    return ranked_tiles[0]                              # nothing safe in top-K -> keep offense
