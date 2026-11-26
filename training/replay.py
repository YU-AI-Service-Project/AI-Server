"""Chronological samples and deterministic station holdout used by WattUp v7 and v9."""

import hashlib
import json

import numpy as np

from observation_builder import NAMES, PROFILES


SEED = 20261119

HORIZONS = range(1, 13)

NAMES_WITH_ETA = NAMES + ["horizonMinutes"]

PROFILE_NAMES = tuple(PROFILES)

PROFILE, TICK, STATION, STEP, CURRENT, LABEL = range(6)

BLOCK = 50000

TIME_SPANS = {
    "train": ("2026-11-11", "2026-11-16"),
    "tune": ("2026-11-16", "2026-11-17"),
    "calibrate": ("2026-11-17", "2026-11-18"),
    "policy": ("2026-11-17", "2026-11-18"),
    "test": ("2026-11-18", "2026-11-20"),
}

PARAMS = dict(
    objective="binary", learning_rate=.04, num_leaves=23, min_data_in_leaf=250,
    lambda_l2=35., seed=SEED, num_threads=2, deterministic=True,
    force_col_wise=True, verbosity=-1, max_bin=127,
)

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()

def station_partition(station):
    return hashlib.sha256(station.encode()).digest()[0] % 2

class Replay:
    def __init__(self, root, holdout):
        self.audit = read_json(root / "data_audit.json")
        self.clock = np.load(root / "clock.npy")
        self.good = np.load(root / "collector_good.npy")
        self.stations = [np.load(root / f"{p}_stations.npy") for p in PROFILE_NAMES]
        self.x = [np.load(root / f"{p}_features.npy", mmap_mode="r") for p in PROFILE_NAMES]
        self.truth = [np.load(root / f"{p}_truth.npy", mmap_mode="r") for p in PROFILE_NAMES]
        self.usable = [np.load(root / f"{p}_usable.npy", mmap_mode="r") for p in PROFILE_NAMES]
        self.held = [np.isin(ids, list(holdout)) for ids in self.stations]
        self.partition = [np.array([station_partition(s) for s in ids]) for ids in self.stations]

    def pairs(self, split, per_train_block):
        rng = np.random.default_rng(SEED)
        start, end = np.array(TIME_SPANS[split], dtype="datetime64[m]")
        all_rows, all_weights = [], []
        pool = transitions = 0
        train = split == "train"
        bad = np.r_[0, np.cumsum(~self.good)]
        for p, valid in enumerate(self.usable):
            station_ok = np.ones(valid.shape[1], dtype=bool) if split == "test" else ~self.held[p]
            if split in ("calibrate", "policy"):
                station_ok &= self.partition[p] == int(split == "policy")
            for step in HORIZONS:
                # A candidate needs a known state at the target and complete collection
                # from its anchor through its target. Respect the split target cutoff.
                valid_ticks = (
                    (self.clock[:-step] >= start) & (self.clock[step:] < end)
                    & (bad[step + 1:] == bad[:-step - 1])
                )
                if not train:
                    valid_ticks &= self.clock[:-step].astype(int) % 30 == 0
                ticks = np.flatnonzero(valid_ticks)
                a, s = np.nonzero(valid[ticks] & (self.truth[p][ticks + step] >= 0) & station_ok)
                t = ticks[a]
                y = self.truth[p][t + step, s]
                current = self.truth[p][t, s]
                if (current < 0).any():
                    raise ValueError("Pair has unknown anchor state")
                change = y != current
                pool += len(y)
                transitions += int(change.sum())
                if train and len(y) > per_train_block:
                    changed = np.flatnonzero(change)
                    stable = np.flatnonzero(~change)
                    n_changed = min(len(changed), max(per_train_block // 3, per_train_block - len(stable)))
                    n_stable = per_train_block - n_changed
                    i = np.sort(np.r_[rng.choice(changed, n_changed, replace=False),
                                      rng.choice(stable, n_stable, replace=False)])
                    weights = np.where(change[i], len(changed) / max(n_changed, 1),
                                       len(stable) / max(n_stable, 1))
                    t, s, current, y = t[i], s[i], current[i], y[i]
                else:
                    weights = np.ones(len(y), dtype=np.float32) if train else None
                rows = np.empty((len(y), 6), dtype=np.int32)
                rows[:, PROFILE], rows[:, STEP] = p, step
                rows[:, TICK], rows[:, STATION], rows[:, CURRENT], rows[:, LABEL] = t, s, current, y
                all_rows.append(rows)
                if train:
                    all_weights.append(weights.astype(np.float32))
        pairs = np.concatenate(all_rows)
        weights = np.concatenate(all_weights) if train else None
        if not len(pairs):
            raise ValueError(f"No evaluable rows in {split}")
        stats = {
            "rows": len(pairs), "available_rate": float(pairs[:, LABEL].mean()),
            "candidate_pairs": pool, "candidate_transitions": transitions,
            "transitions_used": int((pairs[:, CURRENT] != pairs[:, LABEL]).sum()),
            "as_of_first": str(self.clock[pairs[:, TICK].min()]),
            "as_of_last": str(self.clock[pairs[:, TICK].max()]),
            "target_last": str(self.clock[(pairs[:, TICK] + pairs[:, STEP]).max()]),
            "ALL_rows": int((pairs[:, PROFILE] == 0).sum()),
        }
        print("SPLIT", split, stats, flush=True)
        return pairs, weights, stats

    def matrix(self, rows, prior=None):
        result = np.empty((len(rows), len(NAMES_WITH_ETA) + (len(PRIOR_FEATURES) if prior else 0)),
                          dtype=np.float32)
        for start in range(0, len(rows), BLOCK):
            block = rows[start:start + BLOCK]
            features = np.empty((len(block), len(NAMES_WITH_ETA)), dtype=np.float32)
            for p, x in enumerate(self.x):
                mask = block[:, PROFILE] == p
                features[mask, :-1] = x[block[mask, TICK], block[mask, STATION]]
            features[:, -1] = 5 * block[:, STEP]
            result[start:start + len(block)] = (
                append_end_prior(features, NAMES_WITH_ETA, self.clock[block[:, TICK]], prior)
                if prior else features
            )
        return result

    def held_rows(self, rows):
        held = np.zeros(len(rows), dtype=bool)
        for p in range(len(PROFILE_NAMES)):
            mask = rows[:, PROFILE] == p
            held[mask] = self.held[p][rows[mask, STATION]]
        return held
