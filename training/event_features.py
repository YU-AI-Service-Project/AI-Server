"""Recover recent station demand and charger age from observable status events."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


from observation_builder import PROFILES, load_export, prepare_events, timestamp_ns

ROOT = Path(__file__).resolve().parent


EXTRA = [
    "availabilityLosses15", "availabilityGains15", "chargeStarts30", "chargeEnds30",
    "availableAgeMin", "availableAgeKnownFraction",
    "chargingElapsedMin", "recentChargingCount", "longChargingCount",
]


def build(source, replay, output, cutoff=None):
    output.mkdir(parents=True, exist_ok=True)
    c, e, _, audit = prepare_events(*load_export(source))
    metadata = json.loads((replay / "data_audit.json").read_text(encoding="utf-8"))
    with source.open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != metadata["source_zip_sha256"]:
            raise ValueError("Status export differs from the replay source")
    clock = np.load(replay / "clock.npy")
    if cutoff is not None:
        clock = clock[clock <= np.datetime64(cutoff, "m")]
        e = e.loc[e.knownAt.le(pd.Timestamp(cutoff))].reset_index(drop=True)
    tick = timestamp_ns(clock)
    known_at = timestamp_ns(e.knownAt)
    charger = e.chargerIndex.to_numpy(int)
    new_state = e.statCode.to_numpy(int)
    old_state = pd.to_numeric(e.prev_stat, errors="coerce").fillna(0).to_numpy(int)
    provider = timestamp_ns(e.providerAt)
    charging_start = timestamp_ns(e.chargeStart)
    absent = np.iinfo(np.int64).min
    # A provider timestamp can describe state age only after our own acquisition.
    seen_age_start = np.where((provider != absent) & (provider <= known_at), provider, absent)
    lo15 = np.searchsorted(known_at, tick - 15 * 60 * 10**9, side="right")
    lo30 = np.searchsorted(known_at, tick - 30 * 60 * 10**9, side="right")
    stops = np.searchsorted(known_at, tick, side="right")
    public = c.delYn.eq("N") & c.limitYn.eq("N")
    public &= pd.to_numeric(c.lat, errors="coerce").between(-90, 90)
    public &= pd.to_numeric(c.lng, errors="coerce").between(-180, 180)
    real = old_state != 0
    change_masks = [
        real & (old_state == 2) & (new_state != 2),
        real & (old_state != 2) & (new_state == 2),
        real & (old_state != 3) & (new_state == 3),
        real & (old_state == 3) & (new_state != 3),
    ]
    coverage = {}
    for profile, types in PROFILES.items():
        station_ids = np.load(replay / f"{profile}_stations.npy")
        mask = public.to_numpy() & (c.chgerType.isin(types).to_numpy() if types else True)
        if not np.array_equal(np.sort(c.loc[mask, "statId"].unique()), station_ids):
            raise ValueError(f"{profile} station order or membership differs")
        codes = c.statId.map(dict(zip(station_ids, np.arange(len(station_ids))))).fillna(-1).to_numpy(int)
        num_stations = len(station_ids)
        state = np.zeros(len(c), dtype=np.int8)
        state_since = np.full(len(c), absent, dtype=np.int64)
        charging_since = np.full(len(c), absent, dtype=np.int64)
        features = np.lib.format.open_memmap(
            output / f"{profile}_events.npy", mode="w+", dtype=np.float32,
            shape=(len(tick), num_stations, len(EXTRA)),
        )
        cursor = 0
        for t, now in enumerate(tick):
            stop = stops[t]
            for i in range(cursor, stop):
                idx = charger[i]
                state[idx] = new_state[i]
                state_since[idx] = seen_age_start[i]
                charging_since[idx] = charging_start[i]
            cursor = stop
            movements = []
            for low, change in zip((lo15[t], lo15[t], lo30[t], lo30[t]), change_masks):
                ids = charger[low:stop][change[low:stop]]
                movements.append(np.bincount(codes[ids[mask[ids]]], minlength=num_stations))
            active = np.flatnonzero(mask & (state != 0))
            code = codes[active]
            idle = state[active] == 2
            idle_known = idle & (state_since[active] != absent)
            idle_age = (now - np.where(idle_known, state_since[active], now)) / 60e9
            idle_min = np.full(num_stations, np.inf)
            np.minimum.at(idle_min, code[idle_known], np.clip(idle_age[idle_known], 0, 2880))
            idle_total = np.bincount(code[idle], minlength=num_stations)
            idle_known_total = np.bincount(code[idle_known], minlength=num_stations)
            fraction = np.divide(idle_known_total, idle_total, out=np.zeros(num_stations), where=idle_total > 0)
            charging = (state[active] == 3) & (charging_since[active] != absent)
            elapsed = (now - np.where(charging, charging_since[active], now)) / 60e9
            charging &= (elapsed >= 0) & (elapsed <= 2880)
            charge_min = np.full(num_stations, np.inf)
            np.minimum.at(charge_min, code[charging], elapsed[charging])
            recent = np.bincount(code[charging & (elapsed <= 10)], minlength=num_stations)
            long = np.bincount(code[charging & (elapsed >= 90)], minlength=num_stations)
            features[t] = np.column_stack([
                *movements, np.where(np.isfinite(idle_min), idle_min, 0), fraction,
                np.where(np.isfinite(charge_min), charge_min, 0), recent, long,
            ])
        features.flush()
        del features
        coverage[profile] = num_stations
        print("EVENTS", profile, num_stations, len(tick), flush=True)
    (output / "feature_audit.json").write_text(json.dumps({
        "source": metadata["source_zip_sha256"], "ticks": len(tick), "station_counts": coverage,
        "features": EXTRA, "state_event_rows": len(e), "known_time": "Collection completed through asOf",
        "incomplete_runs": audit["incomplete_runs"],
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--replay", type=Path, default=ROOT / "state_replay")
    parser.add_argument("--out", type=Path, default=ROOT / "event_data")
    args = parser.parse_args()
    build(args.source, args.replay, args.out)
