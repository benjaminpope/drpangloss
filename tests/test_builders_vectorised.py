"""The vectorised builders equal the slow per-group loops they replaced.

Each reference below is the loop the builder used to run (one pass over all
samples per group); the builders must return the same arrays, in the same
order and dtype, on small random inputs and on the edge cases.
"""

from types import SimpleNamespace

import numpy as onp
import pytest

from virgil import _closure, gains, observables, oifits
from virgil._utils import group_indices
from tests._shared import PAIRS


def _assert_same(a, b):
    a, b = onp.asarray(a), onp.asarray(b)
    assert a.dtype == b.dtype
    onp.testing.assert_array_equal(a, b)


# -- group_indices, _padded_groups ----------------------------------------


@pytest.mark.parametrize("n", [0, 1, 7, 50])
def test_group_indices_is_one_flatnonzero_per_label(n):
    rng = onp.random.default_rng(n)
    labels = rng.integers(0, 4, n) * 3
    got = group_indices(labels)
    want = [onp.flatnonzero(labels == g) for g in onp.unique(labels)]
    assert len(got) == len(want)
    for a, b in zip(got, want):
        _assert_same(a, b)


@pytest.mark.parametrize("labels", [[], [0], [0, 0, 0], [2, 0, 2, 0, 2]])
def test_padded_groups_matches_the_loop(labels):
    labels = onp.array(labels, dtype=int)
    n_group = int(labels.max()) + 1 if labels.size else 0
    members = [onp.flatnonzero(labels == g) for g in range(n_group)]
    m = max((len(x) for x in members), default=0)
    want = onp.full((n_group, m), 99, dtype=onp.int32)
    for g, x in enumerate(members):
        want[g, : len(x)] = x
    _assert_same(observables._padded_groups(labels, 99), want)


# -- gains.gain_modes ----------------------------------------------------


def _slow_gain_modes(data, telescope, baseline, chromatic):
    """``gain_modes`` with the original per-frame, per-pair passes."""
    built_in = {
        "telescope": telescope,
        "baseline": baseline,
        "chromatic": chromatic,
    }
    index, frame, stations = gains._frames_and_stations(data, True)
    groups, widths, columns = [], [], []

    def add_group(name, width):
        groups.append(name)
        widths.append(float(width))
        return len(groups) - 1

    wavel = onp.broadcast_to(onp.asarray(data.wavel), onp.shape(data.u))[index]
    chrom = (onp.median(wavel) / wavel) ** 2
    ids = {n: add_group(n, w) for n, w in built_in.items() if w is not None}
    for f in onp.unique(frame):
        in_frame = onp.flatnonzero(frame == f)
        pairs = stations[in_frame]
        if "telescope" in ids:
            for t in onp.unique(pairs):
                rows = in_frame[(pairs == t).any(axis=1)]
                columns.append((ids["telescope"], rows, onp.ones(rows.size)))
        for pair in onp.unique(pairs, axis=0):
            rows = in_frame[(pairs == pair).all(axis=1)]
            if "baseline" in ids:
                columns.append((ids["baseline"], rows, onp.ones(rows.size)))
            if "chromatic" in ids:
                columns.append((ids["chromatic"], rows, chrom[rows]))
    return gains._pack(columns, groups, widths, frame)


def _gain_data(frame, stations, wavel, vis_index=None):
    n = len(frame)
    return SimpleNamespace(
        observable_kind="split",
        vis_mat=None,
        frame=onp.asarray(frame),
        stations=onp.asarray(stations),
        u=onp.zeros(n),
        wavel=onp.asarray(wavel),
        vis_index=vis_index,
    )


def _random_gain_data(seed, n_frame, n_wave, vis_index=False):
    rng = onp.random.default_rng(seed)
    # Samples in random order, with frames of unequal size and station pairs
    # that repeat, some of them reversed (the builder sorts each pair).
    rows = []
    for f in range(n_frame):
        for pair in PAIRS[rng.permutation(len(PAIRS))[: rng.integers(1, 7)]]:
            for w in range(n_wave):
                rows.append((f * 2 + 1, *pair[:: rng.choice([1, -1])], w))
    rows = onp.array(rows)[rng.permutation(len(rows))]
    wavel = rng.uniform(2.0e-6, 2.4e-6, n_wave)
    index = None
    if vis_index:
        index = onp.sort(rng.permutation(len(rows))[: 2 * len(rows) // 3])
    return _gain_data(rows[:, 0], rows[:, 1:3], wavel[rows[:, 3]], index)


@pytest.mark.parametrize(
    "seed, n_frame, n_wave, vis_index",
    [
        (0, 1, 1, False),
        (1, 1, 4, False),
        (2, 5, 3, False),
        (3, 4, 2, True),
        (4, 9, 5, False),
    ],
)
@pytest.mark.parametrize(
    "widths",
    [(0.01, None, None), (None, 0.02, 0.03), (0.01, 0.02, 0.03)],
)
def test_gain_modes_equals_the_per_frame_loop(
    seed, n_frame, n_wave, vis_index, widths
):
    data = _random_gain_data(seed, n_frame, n_wave, vis_index)
    new = gains.gain_modes(data, *widths)
    old = _slow_gain_modes(data, *widths)
    assert new.groups == old.groups and new.n_vis == old.n_vis
    for name in ("rows", "shapes", "group", "spanning", "spanning_group"):
        a, b = getattr(new, name), getattr(old, name)
        assert (a is None) == (b is None)
        if a is not None:
            _assert_same(a, b)
    _assert_same(new.widths, old.widths)


# -- _closure ------------------------------------------------------------


def _slow_incidence(group, i1, i2, i3):
    baselines = onp.unique(onp.concatenate([i1[group], i2[group], i3[group]]))
    col = {b: j for j, b in enumerate(baselines)}
    t = onp.zeros((group.size, baselines.size))
    for row, tri in enumerate(group):
        t[row, col[i1[tri]]] += 1.0
        t[row, col[i2[tri]]] += 1.0
        t[row, col[i3[tri]]] -= 1.0
    return t


def _closure_indices(seed, n_frame, n_tel):
    """Closure-phase baseline indices of ``n_tel`` telescopes, shuffled."""
    rng = onp.random.default_rng(seed)
    pairs = [(a, b) for a in range(n_tel) for b in range(a + 1, n_tel)]
    base = {p: k for k, p in enumerate(pairs)}
    tris = [
        (base[a, b], base[b, c], base[a, c])
        for a in range(n_tel)
        for b in range(a + 1, n_tel)
        for c in range(b + 1, n_tel)
    ]
    n_base = len(pairs)
    out = []
    for f in range(n_frame):
        for t in tris:
            out.append([f * n_base + k for k in t])
    out = onp.array(out)[rng.permutation(len(out))]
    return out.T


@pytest.mark.parametrize("n_tel", [3, 4, 5, 6])
@pytest.mark.parametrize("n_frame", [1, 3])
def test_incidence_equals_the_per_row_loop(n_tel, n_frame):
    i1, i2, i3 = _closure_indices(n_tel, n_frame, n_tel)
    for group in _closure._groups(i1, i2, i3):
        _assert_same(
            _closure._incidence(group, i1, i2, i3),
            _slow_incidence(group, i1, i2, i3),
        )


def test_incidence_counts_a_repeated_baseline_twice():
    i1, i2, i3 = onp.array([5]), onp.array([5]), onp.array([9])
    group = onp.array([0])
    _assert_same(
        _closure._incidence(group, i1, i2, i3),
        _slow_incidence(group, i1, i2, i3),
    )


@pytest.mark.parametrize("n_tel", [4, 5])
def test_closure_noise_is_unchanged(n_tel, monkeypatch):
    legs = _closure_indices(7, 3, n_tel)
    new = _closure.ClosureNoise.from_indices(*legs)
    monkeypatch.setattr(_closure, "_incidence", _slow_incidence)
    old = _closure.ClosureNoise.from_indices(*legs)
    for name in new.__dataclass_fields__:
        _assert_same(getattr(new, name), getattr(old, name))


# -- DifferentialPhase.build ---------------------------------------------


def _slow_grids(frame, row, wavel):
    """Per frame: the ``(rows, channels)`` sample indices of the old loops."""
    out = []
    for f in onp.unique(frame):
        in_frame = onp.flatnonzero(frame == f)
        rows = onp.unique(row[in_frame])
        channels = None
        for r in rows:
            w_r = wavel[row == r]
            channels = (
                w_r if channels is None else onp.intersect1d(channels, w_r)
            )
        channels = onp.sort(channels)
        if channels.size == 0:
            continue
        grid = onp.empty((rows.size, channels.size), dtype=int)
        for b, r in enumerate(rows):
            idx = onp.flatnonzero(row == r)
            idx = idx[onp.argsort(wavel[idx])]
            grid[b] = idx[onp.searchsorted(wavel[idx], channels)]
        out.append(grid)
    return out


def _phase_samples(seed, n_frame, drop):
    """Samples of several frames in random order, with channels dropped."""
    rng = onp.random.default_rng(seed)
    channels = onp.linspace(2.0e-6, 2.4e-6, 7)
    rows = []
    for f in range(n_frame):
        for s, pair in enumerate(PAIRS):
            keep = rng.random(channels.size) >= drop
            keep[0] = True
            for w in channels[keep]:
                rows.append((f, s, *pair, w))
    rows = onp.array(rows)[rng.permutation(len(rows))]
    return {
        "frame": rows[:, 0].astype(int),
        "stations": rows[:, 2:4].astype(int),
        "wavel": rows[:, 4],
    }


@pytest.mark.parametrize(
    "seed, n_frame, drop, closure_free, prior_width",
    [
        (0, 1, 0.0, True, None),
        (1, 3, 0.0, True, None),
        (2, 4, 0.1, True, None),
        (3, 3, 0.0, False, None),
        (4, 3, 0.1, True, 0.1),
    ],
)
def test_differential_phase_grid_equals_the_per_row_loops(
    seed, n_frame, drop, closure_free, prior_width
):
    s = _phase_samples(seed, n_frame, drop)
    n = s["frame"].size
    rng = onp.random.default_rng(seed)
    block = observables.DifferentialPhase.build(
        rng.normal(0, 0.1, n),
        onp.full(n, 0.02),
        onp.arange(n),
        s["wavel"],
        s["frame"],
        s["stations"],
        closure_free=closure_free,
        prior_width=prior_width,
        order=1,
    )
    row = observables._labels(onp.column_stack([s["frame"], s["stations"]]))
    want = _slow_grids(s["frame"], row, s["wavel"])
    assert block.grid.shape[0] == len(want)
    for f, grid in enumerate(want):
        nb, nc = grid.shape
        onp.testing.assert_array_equal(block.grid[f, :nb, :nc], grid)
        assert block.chan[f, :nb, :nc].all()
        assert block.chan[f].sum() == nb * nc


def test_differential_phase_one_row_per_frame():
    n, c = 6, 5
    wavel = onp.tile(onp.linspace(2.0e-6, 2.4e-6, c), n)
    frame = onp.repeat(onp.arange(n), c)
    block = observables.DifferentialPhase.build(
        onp.zeros(n * c),
        onp.full(n * c, 0.02),
        onp.arange(n * c),
        wavel,
        frame,
        onp.tile([0, 1], (n * c, 1)),
        closure_free=False,
    )
    assert block.grid.shape[0] == n


# -- oifits._BaselineLookup.times ----------------------------------------


def _slow_times(lookup, frame_mjd):
    """``times`` with the original quadratic comparison of window rows."""
    by_ins = {}
    for start, _, mjd, ins, time in lookup._order:
        by_ins.setdefault(ins, []).append((mjd, time, start))
    for rows in by_ins.values():
        rows.sort()
        for i, (mjd0, t0, a) in enumerate(rows):
            for mjd1, t1, b in rows[i + 1 :]:
                if mjd1 - mjd0 > oifits._MJD_TOLERANCE:
                    break
                if abs(t1 - t0) <= oifits._MJD_TOLERANCE:
                    lookup.link(a, b)
    roots = [lookup._root(start) for start, *_ in lookup._order]
    labels = {root: k for k, root in enumerate(dict.fromkeys(roots))}
    frame = onp.array([labels[root] for root in roots])
    row_mjd = onp.array([mjd for _, _, mjd, _, _ in lookup._order])
    if frame_mjd == "mean":
        sums = onp.bincount(frame, weights=row_mjd)
        row_mjd = (sums / onp.bincount(frame))[frame]
    nwave = [n for _, n, _, _, _ in lookup._order]
    return onp.repeat(row_mjd, nwave), onp.repeat(frame, nwave)


def _lookup(rows, links):
    lookup = oifits._BaselineLookup()
    start = 0
    for ins, mjd, time, nwave in rows:
        lookup.add(ins, (1, 2), mjd, 0.0, start, nwave, time=time)
        start += nwave
    starts = [0]
    for _, _, _, nwave in rows[:-1]:
        starts.append(starts[-1] + nwave)
    for i, j in links:
        lookup.link(starts[i], starts[j])
    return lookup


def _random_rows(seed, n, mode):
    rng = onp.random.default_rng(seed)
    tol = oifits._MJD_TOLERANCE
    if mode == "night":
        # One MJD for the night, snapshots in TIME (the window is everything).
        mjd = onp.full(n, 60000.0)
        time = rng.integers(0, 6, n) * 3.0 * tol + rng.uniform(0, tol / 2, n)
    elif mode == "boundary":
        # Differences at exact multiples of the tolerance, in both columns.
        mjd = 60000.0 + rng.integers(0, 6, n) * tol
        time = rng.integers(0, 6, n) * tol
    else:
        mjd = 60000.0 + rng.uniform(0, 5 * tol, n)
        time = rng.uniform(0, 5 * tol, n)
    ins = rng.integers(0, 2, n)
    nwave = rng.integers(1, 4, n)
    return [
        (("ARR", f"INS{i}"), float(m), float(t), int(w))
        for i, m, t, w in zip(ins, mjd, time, nwave)
    ]


@pytest.mark.parametrize("frame_mjd", ["first", "mean"])
@pytest.mark.parametrize("mode", ["night", "boundary", "random"])
@pytest.mark.parametrize("n", [1, 2, 9, 40])
def test_times_equals_the_pairwise_comparison(n, mode, frame_mjd):
    rows = _random_rows(n, n, mode)
    rng = onp.random.default_rng(n)
    links = [tuple(rng.integers(0, n, 2)) for _ in range(n // 4)]
    new = _lookup(rows, links).times(frame_mjd)
    old = _slow_times(_lookup(rows, links), frame_mjd)
    for a, b in zip(new, old):
        _assert_same(a, b)


def test_times_is_repeatable_and_keeps_its_links():
    rows = _random_rows(3, 20, "random")
    lookup = _lookup(rows, [(0, 5)])
    first = lookup.times("first")
    second = lookup.times("first")
    for a, b in zip(first, second):
        _assert_same(a, b)
