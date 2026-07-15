# What 217 generations of self-play look like on a game that keeps changing

*Ludus research note — 2026-07-08*

Two years ago OpenAI's hide-and-seek agents made a great story: pour agents
into a physics sandbox, let them compete, and watch distinct strategies
*emerge* in sharp phase transitions — box-blocking, ramp use, box-surfing.
I wanted to know what the same recipe looks like on **Boom**, a
Clash-Royale-like lane battler I run as a generational self-play league. This
is an honest account of what I found. It is less cinematic than hide-and-seek,
and the reasons why are the interesting part.

## The setup

Boom is a deterministic JAX game: two players, elixir economy, cards, towers,
three-minute matches. The league is simple generational self-play with gates.
Every ~21 GPU-minutes a **candidate** trains against a pool of past selves,
then it must beat the reigning **champion** over seat-swapped paired matches
with the **lower bound of a 95% confidence interval above 50%** to be
promoted. No CI-clearance, no crown. Every generation is also measured against
two fixed yardsticks — a `random` bot and a scripted `rule_v0` bot — so there
is always a stationary reference.

The twist, which turned out to define everything: **the game kept changing
under the league.** Over the campaign the engine advanced from v8 to v15 as I
chased Clash-Royale mechanics parity — spell flight, charge, building knockback
immunity, a tangential-slide pathing fix, the walk-retarget kiting rule,
king-tower activation, pull spells. The league was never reset. It *resumed
and adapted* across every one of those changes.

## The progression curve is honest, and noisy

![Win rate versus fixed anchors across 217 league generations](/static/blog/img/league_curve.png)

Here is win rate against the two fixed anchors across all 217 generations.
Two things jump out. First, `random` (green) saturates near 100% almost
immediately — it is too weak to be a useful yardstick past gen 5. Second,
`rule_v0` (pink) climbs from ~70% to a plateau around 85%, with a visible step
up around **gen 69** — but it is *noisy and non-monotonic*, dipping to 50% at
points and never becoming a clean upward sweep.

That noise is not a plotting artifact. It is the honest fingerprint of a
**moving target**: each engine change shifted what "good play" means, and the
scripted bot's relative strength moved with it. Explorer generations, which
deliberately play off-policy to find new strategies, add the sharp downward
spikes. If you came expecting the tidy sigmoid of a stationary benchmark, this
is the correction — real training against a changing game and a changing
opponent pool looks like *this*.

## The clean signal is transitivity

Anchor curves are muddy. The sharper question is: **is a later champion
actually better than an earlier one?** I played the current champion
(generation 192) against four of its ancestors, 64 seat-swapped matches each.

![Champion win rate versus ancestor generations, a monotonic gradient](/static/blog/img/league_transitivity.png)

This is the result I am most confident in. The champion beats the ancient
generation-5 policy **80%** of the time, generation 30 **70%**, generation 69
**61%**, and the champion it directly dethroned, generation 166, **55%** — a
clean monotonic gradient. Newer beats older, and it beats *older* older more
decisively. That is exactly the shape genuine, incremental progress should
have. It also tells you the gains are **marginal per step**: promotion gates
cleared at roughly 57%, and 55% over the immediately-prior champion is real
but small. There is no capability explosion here — there is a ladder, climbed
one careful rung at a time.

## The mistake I almost published: "seat asymmetry"

My first read of the head-to-head data was that the champion had a huge
**player-2 advantage** — it won 94–100% of games as player 2 but only 28–59%
as player 1 against the same ancestors. I wrote that down as a finding: Boom
must have a big second-player edge. It was the wrong diagnosis, and catching it
is the most useful thing in this post.

The two seats **train with different decks.** Player 0 gets a 2.6-elixir
*cycle* deck (fast, cheap, Hog-Rider-style); player 1 gets a 3.6-elixir
*beatdown* deck (a Giant, a Wizard, heavier units). A full elixir point apart.
So the "seat" I was measuring was really a *deck*.

The clean way to separate the two is to play the champion against **itself** —
identical policy, so any win-rate skew is 100% attributable to the decks:

| deck setup | player 0 wins | player 1 wins |
|---|---|---|
| cycle (P0) vs beatdown (P1) | **19%** | **72%** |
| mirror — both cycle | 38% | 41% |
| mirror — both beatdown | 44% | 47% |

There it is. The beatdown deck beats the cycle deck **72 to 19** in this
engine; mirror matches are ~50/50 (the residual 3–5 point P0-under-P1 gap is
the *real*, small first-player effect). There is no meaningful seat advantage.
There is a lopsided card matchup, and every "seat asymmetry" number I had was
just measuring which deck each seat happened to hold.

Here is that imbalance in motion — the champion playing *itself*, cycle deck
(blue) versus beatdown deck (red). Same brain on both sides; the red deck
simply wins:

![The beatdown deck defeating the cycle deck, same policy on both sides](/static/blog/img/league_deck_imbalance.gif)

The lesson is a boring, load-bearing one: **if your two seats aren't
symmetric, you are benchmarking the asymmetry, not the agent.** The fix is to
train and evaluate on *mirror* decks (both sides identical), which is the next
change to the league. It is also a small balance verdict on the game itself —
the default beatdown deck is over-tuned against the default cycle deck, and
that wants fixing before either deck anchors a ladder.

## One finding that survived

**Non-transitivity is mostly absent — but not entirely.** In individual matches
the champion occasionally *loses* to gen 5 (I caught one on the first seed I
tried). Over 64 matches that washes out to a clean 80%, but the existence of
rock-paper-scissors pockets is real, and it is a known reason self-play ladders
can stall.

## Why this is less dramatic than hide-and-seek — honestly

Hide-and-seek had two things Boom does not. Its environment was *open-ended* (a
physics sandbox where a genuinely new tool — a ramp, a box — unlocks a new
regime), and it was *stationary* (the rules never moved, so a phase transition
is unmistakable against a fixed backdrop). Boom is a tightly constrained card
game with a fixed action space, and I deliberately kept changing its rules to
chase parity. Under those conditions you should *expect* gradual refinement and
constant re-adaptation, not sharp emergent phase transitions — and that is what
I measured. I cannot honestly claim "the league discovered kiting at gen 166
because I added kiting at v15." What I can claim is narrower and true: the last
two champions were the first trained in the v15-mechanics era, the ladder kept
climbing across five engine changes without a reset, and each rung is a
CI-verified improvement over the last.

## Takeaways

- **Self-play across a moving game works.** The league adapted through v8→v15
  with no resets and kept producing verified champions (30 → 69 → 166 → 192).
- **Trust transitivity over anchor curves** when the game is non-stationary; the
  anchors move, head-to-head does not.
- **Check your seats are symmetric before you trust any number.** I nearly
  published a "second-player advantage" that was really a deck matchup;
  champion-vs-self with mirror decks is what exposed it. Benchmark the agent,
  not the asymmetry.
- **Emergence needs an open-ended, stationary world.** A constrained,
  deliberately-shifting one gives you honest, incremental, hard-won gains
  instead — which is still exactly what you want from a competition ladder.

*Every number here is from the live league's standings file and 64-match
seat-swapped evaluations run on CPU; the figures and the match GIF are
generated from real trajectories, computed without disturbing the GPU training
in progress.*