"""Build candidate-list replay from observed arrival outcomes and proxy ETAs."""

from dataclasses import dataclass

import numpy as np

from replay import CURRENT, LABEL, PROFILE, PROFILE_NAMES, STATION, STEP, TICK


SPANS = {
    "train": ("2026-11-11", "2026-11-16", 15),
    "tune": ("2026-11-16", "2026-11-17", 30),
    "calibration": ("2026-11-17", "2026-11-18", 30),
    "test": ("2026-11-18", "2026-11-20", 30),
}

@dataclass
class Queries:
    rows: np.ndarray
    sizes: np.ndarray
    origins: np.ndarray

    @property
    def starts(self):
        return np.r_[0, np.cumsum(self.sizes[:-1])]

def build_queries(replay, scenarios, split):
    start, end, stride = SPANS[split]
    start, end = np.datetime64(start, "m"), np.datetime64(end, "m")
    ticks = np.flatnonzero((replay.clock >= start) & (replay.clock < end)
                           & (replay.clock.astype(int) % stride == 0))
    bad = np.r_[0, np.cumsum(~replay.good)]
    blocks, sizes, origins = [], [], []
    for p, origin_id, stations, steps in scenarios:
        t = ticks[:, None]
        horizon = steps[(ticks + origin_id) % 2]
        finished = t + horizon
        within = finished < len(replay.clock)
        target_index = np.minimum(finished, len(replay.clock) - 1)
        target = replay.truth[p][target_index, stations]
        current = replay.truth[p][t, stations]
        known = (within & replay.usable[p][t, stations] & (current >= 0) & (target >= 0)
                 & (replay.clock[target_index] < end)
                 & (bad[target_index + 1] == bad[t]))
        if split != "test":
            known &= ~replay.held[p][stations]
        known &= (known.sum(axis=1) >= 2)[:, None]
        ti, si = np.nonzero(known)
        if not len(ti):
            continue
        size = np.count_nonzero(known, axis=1)
        size = size[size > 0]
        blocks.append(np.column_stack([
            np.full(len(ti), p), ticks[ti], stations[si], horizon[ti, si],
            current[ti, si], target[ti, si],
        ]).astype(np.int32))
        sizes.append(size)
        origins.append(np.full(len(size), origin_id, dtype=np.int32))
    queries = Queries(np.concatenate(blocks), np.concatenate(sizes), np.concatenate(origins))
    assert int(queries.sizes.sum()) == len(queries.rows)
    assert (queries.sizes >= 2).all()
    assert np.all(replay.clock[queries.rows[:, TICK] + queries.rows[:, STEP]] < end)
    print("QUERIES", split, len(queries.sizes), len(queries.rows), flush=True)
    return queries

def first_choice(data, scores):
    group = np.repeat(np.arange(len(data.sizes)), data.sizes)
    order = np.lexsort((data.rows[:, STATION], data.rows[:, STEP], -scores, group))
    return order[data.starts]

def constrained_queries(data, replay):
    """Select hard candidate sets using only information known before departure."""
    source_rows = data.rows
    group = np.repeat(np.arange(len(data.sizes)), data.sizes)
    chargers = np.empty(len(source_rows))
    for p in range(len(PROFILE_NAMES)):
        mask = source_rows[:, PROFILE] == p
        chargers[mask] = replay.x[p][source_rows[mask, TICK], source_rows[mask, STATION], 1]
    busy = source_rows[:, CURRENT] == 0
    last = chargers == 1
    counts = np.bincount(group[busy], minlength=len(data.sizes))
    present = np.logical_or.reduceat(last, data.starts)
    nearest_last = first_choice(data, np.where(last, -source_rows[:, STEP].astype(float), -np.inf))
    chosen_open = np.zeros(len(source_rows), dtype=bool)
    chosen_open[nearest_last[present]] = True
    restricted = {
        "busy_only": (busy, counts >= 2),
        "last_slot_mixed": (busy | chosen_open, (counts >= 1) & present),
    }
    results = {"routine": data}
    for name, (row_mask, group_mask) in restricted.items():
        keep = row_mask & group_mask[group]
        sizes = np.bincount(group[keep], minlength=len(data.sizes))[group_mask]
        results[name] = Queries(source_rows[keep], sizes, data.origins[group_mask])
        print("CONSTRAINED", name, len(sizes), int(sizes.sum()), flush=True)
    return results
