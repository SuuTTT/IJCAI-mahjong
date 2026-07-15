"""Hand-checked validation of the Leduc engine: legal actions, betting math,
showdown resolution, pot payoffs. Run: python3 test_leduc.py"""
import leduc as L

ok = 0
fail = 0


def check(name, cond):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name}")
    else:
        fail += 1
        print(f"  FAIL  {name}")


# ---- legal actions at the start ----
s = L.new_game(2, 0)  # p0=K, p1=J
check("start: no bet outstanding -> {call, raise}, no fold",
      set(L.legal_actions(s)) == {L.CALL, L.RAISE})

# ---- check-check ends round 0 -> chance_pub ----
s1 = L.apply_action(s, L.CALL)   # p0 checks
check("after p0 check: p1 to act, round 0",
      s1.kind == "decision" and s1.to_act == 1 and s1.round == 0)
s2 = L.apply_action(s1, L.CALL)  # p1 checks
check("check-check -> public chance node", s2.kind == "chance_pub")
check("check-check: contribs still ante 1/1", s2.c0 == 1 and s2.c1 == 1)

# ---- bet then fold: pot math ----
s = L.new_game(1, 2)             # p0=Q, p1=K
sb = L.apply_action(s, L.RAISE)  # p0 bets 2 -> c0=3
check("p0 bet round0: c0=3 (ante1+2), c1=1", sb.c0 == 3 and sb.c1 == 1)
check("facing bet: p1 has {fold, call, raise}",
      set(L.legal_actions(sb)) == {L.FOLD, L.CALL, L.RAISE})
sf = L.apply_action(sb, L.FOLD)  # p1 folds
check("p1 folds -> terminal, winner p0", sf.kind == "terminal" and sf.winner == 0)
check("p0 wins folder's contribution (=1): payoff_p0 = +1",
      L.payoff_p0(sf) == 1)

# ---- bet-raise-call round0, then showdown pairing ----
s = L.new_game(0, 1)                  # p0=J, p1=Q
s = L.apply_action(s, L.RAISE)        # p0 bet 2 -> c0=3
s = L.apply_action(s, L.RAISE)        # p1 raise -> c1 = high(3)+2 = 5
check("p1 raise: c1=5, raises=2", s.c1 == 5 and s.raises == 2)
check("after 2 raises: no more raise for p0",
      set(L.legal_actions(s)) == {L.FOLD, L.CALL})
s = L.apply_action(s, L.CALL)         # p0 calls -> c0=5
check("bet-raise-call -> public chance, contribs 5/5",
      s.kind == "chance_pub" and s.c0 == 5 and s.c1 == 5)

# public children: p0=J,p1=Q dealt -> remaining {J,K,K,Q} = ranks J:1,Q:1,K:2
kids = L.public_children(s)
ranks = {r: p for r, p, _ in kids}
check("public dist after J,Q dealt: P(J)=1/4, P(Q)=1/4, P(K)=2/4",
      abs(ranks[0] - 0.25) < 1e-9 and abs(ranks[1] - 0.25) < 1e-9
      and abs(ranks[2] - 0.5) < 1e-9)

# take public = Q -> p1 pairs -> p1 should win showdown if both check round1
sQ = [ns for r, p, ns in kids if r == 1][0]
check("public=Q, round 1, p0 to act", sQ.pub == 1 and sQ.round == 1 and sQ.to_act == 0)
r1a = L.apply_action(sQ, L.CALL)      # p0 check
r1b = L.apply_action(r1a, L.CALL)     # p1 check -> showdown
check("round1 check-check -> terminal showdown", r1b.kind == "terminal")
check("p1 pairs Q -> p1 wins, payoff_p0 = -5", r1b.winner == 1 and L.payoff_p0(r1b) == -5)

# ---- showdown high-card and split ----
# p0=K,p1=J, public=Q (no pair): K>J -> p0 wins
check("high card K beats J", L._showdown_winner(2, 0, 1) == 0)
check("pair beats high card (p1 pairs)", L._showdown_winner(2, 1, 1) == 1)
check("equal ranks -> split", L._showdown_winner(1, 1, 2) == -1)

# ---- round1 bet size is 4 ----
s = L.new_game(2, 0)
s = L.apply_action(s, L.CALL); s = L.apply_action(s, L.CALL)  # check-check
sQ = L.public_children(s)[0][2]      # some public
sbet = L.apply_action(sQ, L.RAISE)   # round1 bet
me = sbet  # p1 to act now, p0 committed ante1 + 4
check("round1 bet size 4: bettor contrib = 1+4 = 5",
      max(sbet.c0, sbet.c1) == 5)

# ---- infoset hides opponent card / distinguishes actors ----
sa = L.new_game(2, 0)                 # p0=K to act at start
sb2 = L.apply_action(sa, L.CALL)      # p1=J to act after p0 check
ka = L.infoset_key(sa)
kb = L.infoset_key(sb2)
check("p0-start infoset key uses p0 private (K=2)", ka[0] == 2)
check("p1-after-check infoset uses p1 private (J=0)", kb[0] == 0)
check("the two infosets differ (distinct histories)", ka != kb)

# ---- full enumeration sanity ----
iss = L.enumerate_infosets()
print(f"  [info] total decision infosets = {len(iss)}")
check("infoset count in plausible Leduc range (100-1000)", 100 <= len(iss) <= 1000)

print(f"\n{ok} passed, {fail} failed")
import sys
sys.exit(1 if fail else 0)
