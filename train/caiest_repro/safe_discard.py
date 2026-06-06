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

def choose_discard(agent, ranked_tiles, top_k=TOP_K):
    """ranked_tiles: legal discard tiles, policy order (best first). Return the discard to play."""
    hist = agent.history; packs = agent.packs
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
