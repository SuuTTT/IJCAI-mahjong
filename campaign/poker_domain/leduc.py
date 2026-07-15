"""Leduc Hold'em engine (2-player, limit, standard rules).

Deck: 6 cards = 2 suits x 3 ranks {J=0, Q=1, K=2}.
Each player antes 1, is dealt 1 private card. Round 0 betting, then 1 public
card is revealed, then round 1 betting, then showdown.
Bet size: 2 in round 0, 4 in round 1. Max 2 raises per round.
Showdown: pairing the public card wins; else higher private rank; equal = split.

Actions:  0 = fold,  1 = call/check,  2 = raise/bet.
Player 0 acts first in both betting rounds.

State is an immutable dict-like namedtuple. All payoffs are to player 0
(zero-sum: player 1's payoff = -player 0's).
"""
from collections import namedtuple

RANKS = 3                      # J, Q, K
RANK_NAMES = ["J", "Q", "K"]
DECK = (0, 0, 1, 1, 2, 2)      # physical cards (rank of each of the 6 cards)
ANTE = 1
BET = (2, 4)                   # raise size per round
MAX_RAISES = 2

FOLD, CALL, RAISE = 0, 1, 2
ACTIONS = (FOLD, CALL, RAISE)
NACT = 3

# state fields
State = namedtuple("State", [
    "round",       # 0 or 1
    "to_act",      # 0 or 1 (only meaningful for decision nodes)
    "c0", "c1",    # chips committed by each player (includes ante)
    "raises",      # raises made this round
    "r0", "r1",    # tuples of actions taken in round 0 / round 1
    "p0", "p1",    # private ranks
    "pub",         # public rank or -1
    "kind",        # "decision", "chance_pub", "terminal"
    "winner",      # for terminal: 0, 1, or -1 (split); None otherwise
    "folder",      # player who folded, or None
])


def new_game(p0, p1):
    """Root state after private deal (p0, p1 are ranks). Round 0, player 0 acts."""
    return State(round=0, to_act=0, c0=ANTE, c1=ANTE, raises=0,
                 r0=(), r1=(), p0=p0, p1=p1, pub=-1,
                 kind="decision", winner=None, folder=None)


def legal_actions(s):
    assert s.kind == "decision"
    c = (s.c0, s.c1)
    high = max(c)
    facing = high > c[s.to_act]
    acts = []
    if facing:
        acts.append(FOLD)
    acts.append(CALL)
    if s.raises < MAX_RAISES:
        acts.append(RAISE)
    return acts


def _hist(s):
    return s.r0 if s.round == 0 else s.r1


def _showdown_winner(p0, p1, pub):
    p0pair = (p0 == pub)
    p1pair = (p1 == pub)
    if p0pair and not p1pair:
        return 0
    if p1pair and not p0pair:
        return 1
    if p0 > p1:
        return 0
    if p1 > p0:
        return 1
    return -1  # split


def apply_action(s, a):
    """Return the successor state after action a at decision node s.
    The successor may be a decision, a chance_pub node, or terminal."""
    assert s.kind == "decision"
    assert a in legal_actions(s), f"illegal action {a} in {s}"
    c = [s.c0, s.c1]
    me = s.to_act
    opp = 1 - me
    high = max(c)
    B = BET[s.round]
    hist = _hist(s) + (a,)
    raises = s.raises

    if a == FOLD:
        winner = opp
        ns = s._replace(kind="terminal", winner=winner, folder=me,
                        r0=hist if s.round == 0 else s.r0,
                        r1=hist if s.round == 1 else s.r1)
        return ns

    if a == CALL:
        c[me] = high                      # match (no-op if it was a check)
    else:  # RAISE
        c[me] = high + B
        raises += 1

    round_over = (c[0] == c[1] and len(hist) >= 2)

    base = dict(c0=c[0], c1=c[1])
    if s.round == 0:
        base["r0"] = hist
    else:
        base["r1"] = hist

    if not round_over:
        return s._replace(to_act=opp, raises=raises, **base)

    # round is over
    if s.round == 0:
        # go to public-card chance node
        return s._replace(round=0, kind="chance_pub", raises=raises, **base)
    else:
        # showdown
        w = _showdown_winner(s.p0, s.p1, s.pub)
        return s._replace(kind="terminal", winner=w, folder=None, raises=raises,
                          **base)


def public_children(s):
    """For a chance_pub node: return list of (rank, prob, next_state).
    Enumerates remaining physical cards (the 4 not dealt to players)."""
    assert s.kind == "chance_pub"
    remaining = list(DECK)
    # remove one physical copy for each private card
    remaining.remove(s.p0)
    remaining.remove(s.p1)
    n = len(remaining)  # 4
    out = []
    # group by rank for probabilities
    from collections import Counter
    cnt = Counter(remaining)
    for rank, k in sorted(cnt.items()):
        prob = k / n
        ns = s._replace(round=1, to_act=0, raises=0, pub=rank,
                        kind="decision")
        out.append((rank, prob, ns))
    return out


def sample_public(s, rng):
    remaining = list(DECK)
    remaining.remove(s.p0)
    remaining.remove(s.p1)
    rank = remaining[rng.randrange(len(remaining))]
    return s._replace(round=1, to_act=0, raises=0, pub=rank, kind="decision")


def payoff_p0(s):
    """Terminal payoff to player 0."""
    assert s.kind == "terminal"
    if s.folder is not None:
        # winner takes folder's contribution
        if s.folder == 0:
            return -s.c0
        else:
            return s.c1
    # showdown: contribs equal
    if s.winner == 0:
        return s.c1
    elif s.winner == 1:
        return -s.c0
    return 0  # split


# ---------- infoset ----------

def infoset_key(s):
    """Key for the acting player's infoset at a decision node."""
    assert s.kind == "decision"
    priv = s.p0 if s.to_act == 0 else s.p1
    return (priv, s.pub, s.round, s.r0, s.r1)


def enumerate_infosets():
    """Return dict: infoset_key -> sorted list of legal actions.
    Enumerated over the full game tree (all deals + betting lines)."""
    out = {}

    def rec(s):
        if s.kind == "terminal":
            return
        if s.kind == "chance_pub":
            for _, _, ns in public_children(s):
                rec(ns)
            return
        k = infoset_key(s)
        la = legal_actions(s)
        if k not in out:
            out[k] = la
        else:
            assert out[k] == la, f"legal-action mismatch for {k}"
        for a in la:
            rec(apply_action(s, a))

    # all ordered private deals (physical), dedup by rank pair is fine but we
    # enumerate rank pairs consistent with the deck (need 2 distinct physical).
    for p0 in range(RANKS):
        for p1 in range(RANKS):
            # feasible if enough physical copies: same rank needs 2 copies (ok)
            if p0 == p1:
                # need two copies of that rank -> available (2 each)
                pass
            rec(new_game(p0, p1))
    return out
