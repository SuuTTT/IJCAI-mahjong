import numpy as np

from boom.cards import ARCHETYPES, CARDS, CARD_NAMES, DECK_A, DECK_B, load_cards


def test_table_loads_and_validates():
    t, names, refs = load_cards()
    assert len(names) == 61
    assert len(set(names)) == 61, "card names must be unique"


def test_archetype_coverage():
    counts = np.bincount(CARDS.archetype, minlength=len(ARCHETYPES))
    assert (counts >= 2).all(), dict(zip(ARCHETYPES, counts))


def test_decks_are_valid():
    for deck in (DECK_A, DECK_B):
        assert deck.shape == (8,)
        assert len(set(deck.tolist())) == 8
        assert ((deck >= 0) & (deck < 60)).all()


def test_units_have_sane_stats():
    units = CARDS.is_spell == 0
    # kamikazes attack exactly once — sustained dps is meaningless for them
    att = units & (CARDS.dmg > 0) & (CARDS.suicide == 0)
    # source-data envelope: per-body dps stays within the analogs' actual spread
    # (lava-hound-like ~45 up to mini-pekka-like ~450)
    dps = CARDS.dmg * 5 / np.maximum(CARDS.period, 1)
    assert (dps[att] >= 20).all() and (dps[att] <= 460).all(), dps[att]
    # spells and buildings are stationary
    assert (CARDS.speed[CARDS.is_spell == 1] == 0).all()
    assert (CARDS.speed[CARDS.archetype == ARCHETYPES.index("building")] == 0).all()
