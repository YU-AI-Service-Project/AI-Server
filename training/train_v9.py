"""Fit ETA availability models on clean replay rows and measure recommendation tradeoffs."""

import argparse
import json
from pathlib import Path
import sys
import zipfile

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.neighbors import BallTree

from event_features import EXTRA
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from probability_curve import calibration_input, probabilities  # noqa: E402
from replay import (  # noqa: E402
    CURRENT, LABEL, NAMES_WITH_ETA, PARAMS, PROFILE, PROFILE_NAMES, Replay, STATION, STEP, TICK,
    read_json, sha,
)
from query_replay import build_queries, constrained_queries  # noqa: E402


BLOCK = 50000
AVAILABLE = NAMES_WITH_ETA.index("availableCount")
MINUTES = range(5, 61, 5)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


class EventReplay(Replay):
    def __init__(self, state_dir, events, holdout):
        super().__init__(state_dir, holdout)
        event_audit = read_json(events / "feature_audit.json")
        if event_audit["source"] != self.audit["source_zip_sha256"] or event_audit["features"] != EXTRA:
            raise ValueError("Status replay and event features do not match")
        self.events = [np.load(events / f"{p}_events.npy", mmap_mode="r") for p in PROFILE_NAMES]
        for p, extra in enumerate(self.events):
            if extra.shape != (*self.x[p].shape[:2], len(EXTRA)):
                raise ValueError("Event features have an incompatible shape")

    def additional(self, rows):
        values = np.empty((len(rows), len(EXTRA)), dtype=np.float32)
        for p, extra in enumerate(self.events):
            match = rows[:, PROFILE] == p
            values[match] = extra[rows[match, TICK], rows[match, STATION]]
        return values

    def design(self, rows, extended):
        core = self.matrix(rows)
        return np.column_stack([core[:, :-1], self.additional(rows), core[:, -1]]) if extended else core

    def available(self, rows):
        values = np.empty(len(rows), dtype=np.int16)
        for p, array in enumerate(self.x):
            mask = rows[:, PROFILE] == p
            values[mask] = array[rows[mask, TICK], rows[mask, STATION], AVAILABLE]
        return values


def raw_predictions(model, replay, rows, extended):
    output = np.empty(len(rows))
    for a in range(0, len(rows), BLOCK):
        end = min(a + BLOCK, len(rows))
        matrix = replay.design(rows[a:end], extended)
        output[a:end] = model.predict(matrix, raw_score=True, num_threads=2)
    return output


def fit_model(replay, training, tune, weights, extended):
    names = NAMES_WITH_ETA[:-1] + (EXTRA if extended else []) + NAMES_WITH_ETA[-1:]
    data = lgb.Dataset(replay.design(training, extended), label=training[:, LABEL],
                       feature_name=names, weight=weights / weights.mean())
    validation = lgb.Dataset(replay.design(tune, extended), label=tune[:, LABEL], reference=data)
    return lgb.train(PARAMS, data, num_boost_round=300, valid_sets=[validation],
                     callbacks=[lgb.early_stopping(25, verbose=False)])


def fit_calibrators(rows, score):
    current = rows[:, CURRENT]
    values = {}
    for state in (0, 1):
        mask = current == state
        x = calibration_input(score[mask], 5 * rows[mask, STEP])
        fit = LogisticRegression(C=1., max_iter=750, random_state=20261126).fit(x, rows[mask, LABEL])
        values[str(state)] = {"coefficients": fit.coef_[0].tolist(), "intercept": float(fit.intercept_[0])}
    return values


def predict_new(model, calibrators, replay, rows, extended):
    score = raw_predictions(model, replay, rows, extended)
    return probabilities(score, 5 * rows[:, STEP], rows[:, CURRENT], calibrators)


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def test_metrics(rows, probability, available):
    observed = rows[:, LABEL]
    scope = {
        "ALL": rows[:, PROFILE] == 0,
        "ALL_last_slot": (rows[:, PROFILE] == 0) & (available == 1),
        "ALL_currently_unavailable": (rows[:, PROFILE] == 0) & (available == 0),
    }
    scope.update({f"ALL_{h}min": scope["ALL"] & (rows[:, STEP] * 5 == h) for h in MINUTES})
    result = {}
    for name, mask in scope.items():
        y, p = observed[mask], probability[mask]
        result[name] = {"rows": len(y), "brier": brier(p, y), "actual_rate": float(y.mean()),
                        "mean_prediction": float(p.mean())}
        if name == "ALL_last_slot":
            result[name]["loss_average_precision"] = float(average_precision_score(1 - y, 1 - p))
    return result


def closest_templates(source, replay):
    with zipfile.ZipFile(source) as archive, archive.open("export/charger.csv") as file:
        c = pd.read_csv(file, usecols=["statId", "lat", "lng", "limitYn", "delYn"], dtype={"statId": str})
    c = c[c.limitYn.eq("N") & c.delYn.eq("N") & c.lat.between(35, 36.5) & c.lng.between(127.9, 129.2)]
    points = c.groupby("statId")[["lat", "lng"]].median().reindex(replay.stations[0]).to_numpy()
    located = np.flatnonzero(np.isfinite(points).all(axis=1))
    tree = BallTree(np.radians(points[located]), metric="haversine")
    bounds = np.nanmin(points, axis=0), np.nanmax(points, axis=0)
    grid = np.stack(np.meshgrid(*[np.arange(lo, hi + .012, .024) + .009
                                   for lo, hi in zip(*bounds)]), axis=-1).reshape(-1, 2)
    radians, neighbor = tree.query(np.radians(grid), k=10)
    scenarios = []
    distance = {}
    for site, (km, j) in enumerate(zip(6371 * radians, neighbor)):
        near = km <= 11
        if near.sum() < 4:
            continue
        stations = located[j[near]]
        km = km[near] * 1.25
        eta_min = 4 + 60 * km[None, :] / np.array([20, 32])[:, None]
        steps = np.minimum(np.ceil(eta_min / 5), 12).astype(int)
        scenarios.append((0, site, stations, steps))
        for station, d in zip(stations, km):
            distance[site, int(station)] = float(d)
    return scenarios, distance


def rank_value(data, prediction, distance, mode):
    if mode == "distance":
        primary, secondary = distance, -prediction
    elif mode == "probability":
        primary, secondary = -prediction, distance
    else:
        primary, secondary = 2 * distance + 15 * (1 - prediction), distance
    group = np.repeat(np.arange(len(data.sizes)), data.sizes)
    order = np.lexsort((data.rows[:, STATION], secondary, primary, group))
    return order[data.starts]


def compare_policy(data, prediction, distance):
    outcome = data.rows[:, LABEL]
    options = np.logical_or.reduceat(outcome == 1, data.starts)
    if not len(options):
        return {}
    best = {}
    for mode in ("distance", "probability", "balanced"):
        top = rank_value(data, prediction, distance, mode)
        best[mode] = {
            "queries": len(top),
            "available_first_choices": int(outcome[top].sum()),
            "arrival_success": float(outcome[top].mean()),
            "mean_distance_km": float(distance[top].mean()),
            "mean_eta_minutes": float(5 * data.rows[top, STEP].mean()),
            "would_be_available_among_solvable": float(outcome[top[options]].mean()) if options.any() else None,
        }
    return best


def compare_difficult_batches(replay, queries, distances, model, calibrators, extended):
    constrained = constrained_queries(queries, replay)
    results = {}
    for name in ("busy_only", "last_slot_mixed"):
        subset = constrained[name]
        if len(subset.rows):
            sites = np.repeat(subset.origins, subset.sizes)
            distance = np.fromiter(
                (distances[int(o), int(s)] for o, s in zip(sites, subset.rows[:, STATION])),
                dtype=float, count=len(subset.rows),
            )
            score = predict_new(model, calibrators, replay, subset.rows, extended)
            results[name] = compare_policy(subset, score, distance)
    return results


def make_fixtures(replay, data, distance, prediction, names, version, out):
    cases = []
    for i in np.unique(np.linspace(0, len(data.sizes) - 1, 36).astype(int)):
        start = data.starts[i]
        end = start + data.sizes[i]
        group = data.rows[start:end]
        asof = str(replay.clock[group[0, TICK]]) + ":00+09:00"
        items = []
        for row, feature, km in zip(group, replay.design(group, names[-2] == EXTRA[-1]), distance[start:end]):
            p, _, s, h, _, _ = row
            items.append({
                "requestItemId": str(len(items)), "stationId": str(replay.stations[p][s]),
                "profile": PROFILE_NAMES[p], "asOf": asof, "etaSeconds": 300 * int(h),
                "roadDistanceM": float(1000 * km),
                "features": dict(zip(names[:-1], feature[:-1].astype(float))) | {
                    "coverageLast30m": 1., "historyKnown30m": 1.,
                },
            })
        cases.append({"request": {"requestId": f"v9-replay-{i}", "requestAt": asof,
                                  "featureVersion": version, "items": items},
                      "expected_probabilities": prediction[start:end].tolist()})
    save(out / "verification_fixtures.json", cases)
    save(out.parent / "example_ai_request.json", cases[0]["request"])


def run(source, out):
    out.mkdir(parents=True, exist_ok=True)
    reference = ROOT / "reference"
    old_card = read_json(reference / "model_card.json")
    old_model = lgb.Booster(model_file=str(reference / old_card["model_file"]))
    old_cal = old_card["probability"]
    holdout = read_json(reference / "station_holdout.json")
    replay = EventReplay(ROOT / "state_replay", ROOT / "event_data", set(holdout))
    source_hash = sha(source)
    if source_hash != replay.audit["source_zip_sha256"]:
        raise ValueError("Mismatched charging state source")

    train, weights, a = replay.pairs("train", 25000)
    tune, _, b = replay.pairs("tune", 25000)
    calibration, _, c = replay.pairs("calibrate", 25000)
    assert a["target_last"] < b["as_of_first"] and b["target_last"] < c["as_of_first"]
    all_profile = tune[:, PROFILE] == 0
    last_slot = all_profile & (replay.available(tune) == 1)
    candidates, fits = {}, {}
    for label, extended in (("core_refit", False), ("event_enhanced", True)):
        print("FIT", label, len(train), flush=True)
        model = fit_model(replay, train, tune, weights, extended)
        p = 1 / (1 + np.exp(-raw_predictions(model, replay, tune, extended)))
        all_loss = brier(p[all_profile], tune[all_profile, LABEL])
        slot_loss = brier(p[last_slot], tune[last_slot, LABEL])
        candidates[label] = {"ALL_brier": all_loss, "last_slot_brier": slot_loss, "selection_loss": .7*all_loss + .3*slot_loss,
                             "best_iteration": model.best_iteration}
        fits[label] = (model, extended)
        print("VALIDATED", label, candidates[label], flush=True)
    winner = min(candidates, key=lambda name: candidates[name]["selection_loss"])
    picked, selected_extended = fits[winner]
    calibrators = fit_calibrators(calibration, raw_predictions(picked, replay, calibration, selected_extended))
    picked.save_model(str(out / f"{winner}.txt"))
    card = {
        "model_version": "wattup-eta-v9-20261126",
        "selected": winner, "model_file": f"{winner}.txt", "features": picked.feature_name(),
        "feature_version": "daegu-eta-v9-events-37" if selected_extended else old_card["feature_version"],
        "profiles": old_card["profiles"], "max_state_age_seconds": 450,
        "calibrators": calibrators, "production_ready": False,
    }
    save(out / "model_card.json", card)
    save(out / "selection_lock.json", {
        "data_sha256": source_hash, "selected": winner,
        "model_sha256": sha(out / card["model_file"]), "card_sha256": sha(out / "model_card.json"),
        "candidates": candidates, "selection": "70% overall ALL Brier and 30% last-slot ALL Brier on 16 Nov; raw scores",
    })
    print("LOCKED", winner, flush=True)
    del train, tune, weights, calibration
    test, _, d = replay.pairs("test", 25000)
    assert c["target_last"] < d["as_of_first"]
    available = replay.available(test)
    old_raw = raw_predictions(old_model, replay, test, False)
    old_p = 1 / (1 + np.exp(-np.clip(old_cal["score_multiplier"] * old_raw + old_cal["intercept"], -700, 700)))
    v9_p = predict_new(picked, calibrators, replay, test, selected_extended)
    metrics = {"v8": test_metrics(test, old_p, available), "v9": test_metrics(test, v9_p, available)}
    print("TEST_BRIER", {model: value["ALL"]["brier"] for model, value in metrics.items()}, flush=True)
    del test, old_raw, old_p, v9_p

    scenarios, distances = closest_templates(source, replay)
    queries = build_queries(replay, scenarios, "test")
    origins = np.repeat(queries.origins, queries.sizes)
    distance = np.fromiter((distances[int(o), int(s)] for o, s in zip(origins, queries.rows[:, STATION])),
                           dtype=float, count=len(queries.rows))
    assert queries.sizes.max() <= 10
    query_p = predict_new(picked, calibrators, replay, queries.rows, selected_extended)
    policies = compare_policy(queries, query_p, distance)
    old_score = raw_predictions(old_model, replay, queries.rows, False)
    query_old = 1 / (1 + np.exp(-np.clip(old_cal["score_multiplier"] * old_score + old_cal["intercept"], -700, 700)))
    locations = {"v8": compare_policy(queries, query_old, distance), "v9": policies}
    save(out / "metrics.json", {"model": metrics, "recommendation": locations,
                                "recommendation_stress": compare_difficult_batches(
                                    replay, queries, distances, picked, calibrators, selected_extended,
                                ),
                                "splits": {"train": a, "tune": b, "calibration": c, "test": d},
                                "maximum_candidates": int(queries.sizes.max()),
                                "route_evaluation": "Proxy km from geodesic distance x1.25; replay ETA derived from distance",
                                "test_status": "18–19 Nov, reused held-out dates for prior model iterations; not a fresh field study"})
    make_fixtures(replay, queries, distance, query_p, card["features"], card["feature_version"], out)
    print("POLICY", policies, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "retrained" / "results")
    args = parser.parse_args()
    run(args.source, args.out)
