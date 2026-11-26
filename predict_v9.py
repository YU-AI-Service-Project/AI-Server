"""Predict charging availability for each candidate at its own arrival."""

from datetime import datetime, timedelta
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np

from probability_curve import probabilities


def timestamp(value):
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError):
        raise ValueError("INVALID_TIMESTAMP") from None
    if moment.utcoffset() is None:
        raise ValueError("TIMEZONE_OFFSET_REQUIRED")
    return moment


def forecast_grid(as_of, request_time, eta_seconds, max_age):
    observed = timestamp(as_of)
    age = (request_time - observed).total_seconds()
    if age < 0:
        raise ValueError("FEATURE_TIME_AFTER_REQUEST")
    if age > max_age:
        raise ValueError("STALE_FEATURES")
    if isinstance(eta_seconds, bool) or not isinstance(eta_seconds, (int, float)) or not math.isfinite(eta_seconds) or eta_seconds < 0:
        raise ValueError("INVALID_ETA_SECONDS")
    lead = age + eta_seconds
    if lead > 3600:
        raise ValueError("HORIZON_UNSUPPORTED")
    upper = max(5, 5 * math.ceil(lead / 300))
    lower = max(5, 5 * math.floor(lead / 300))
    fraction = (lead / 60 - lower) / (upper - lower) if upper != lower else 0.
    arrival = request_time + timedelta(seconds=eta_seconds)
    forecast_start = observed + timedelta(minutes=lower)
    forecast_end = observed + timedelta(minutes=upper)
    timing = {
        "asOf": observed.isoformat(), "requestAt": request_time.isoformat(),
        "requestedArrivalAt": arrival.isoformat(), "forecastAt": forecast_end.isoformat(),
        "forecastWindowStart": forecast_start.isoformat(),
        "horizonMinutes": upper, "etaSeconds": float(eta_seconds),
        "forecastRoundingSeconds": (forecast_end - arrival).total_seconds(),
        "featureAgeSeconds": age,
    }
    return lower, upper, fraction, timing


class ETAPredictorV9:
    def __init__(self, model_dir):
        root = Path(model_dir)
        self.card = json.loads((root / "model_card.json").read_text(encoding="utf-8"))
        self.model = lgb.Booster(model_file=str(root / self.card["model_file"]))
        if self.model.feature_name() != self.card["features"] or self.card["features"][-1] != "horizonMinutes":
            raise ValueError("MODEL_FEATURE_SCHEMA_MISMATCH")
        self.input_names = self.card["features"][:-1]

    def core_values(self, features, profile):
        raw = [features.get(name) for name in [*self.input_names, "coverageLast30m", "historyKnown30m"]]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in raw):
            raise ValueError("INPUT_INCOMPLETE")
        if raw[-2:] != [1, 1] or self.card["profiles"].get(profile, -1) != features["profileIndex"]:
            raise ValueError("INPUT_INCOMPLETE")
        total, available, charging, other, unknown = [
            features[name] for name in ("totalCount", "availableCount", "chargingCount", "otherCount", "unknownCount")
        ]
        if (any(value < 0 or value % 1 for value in (total, available, charging, other, unknown))
                or total == 0 or unknown != 0 or total != available + charging + other):
            raise ValueError("INPUT_INCOMPLETE")
        return raw[:-2]

    def predict_batch(self, request):
        if request.get("featureVersion") != self.card["feature_version"]:
            raise ValueError("FEATURE_VERSION_MISMATCH")
        items = request.get("items", [])
        if not 1 <= len(items) <= 10:
            raise ValueError("BATCH_SIZE_OUT_OF_RANGE")
        ids = [item.get("requestItemId") for item in items]
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("INVALID_OR_DUPLICATE_REQUEST_ITEM_ID")
        requested = timestamp(request["requestAt"])
        output, matrix, scores = [], [], []
        for item in items:
            station = item["stationId"]
            entry = {
                "requestItemId": item["requestItemId"],
                "stationId": station if station.startswith("KECO:") else f"KECO:{station}",
                "profile": item["profile"],
                "etaGeneratedAt": item.get("etaGeneratedAt"), "roadDistanceM": item.get("roadDistanceM"),
                "status": "OK", "probability": None,
                "modelVersion": self.card["model_version"], "featureVersion": self.card["feature_version"],
            }
            output.append(entry)
            try:
                lower, upper, fraction, timing = forecast_grid(
                    item["asOf"], requested, item["etaSeconds"], self.card["max_state_age_seconds"],
                )
                core = self.core_values(item["features"], item["profile"])
            except ValueError as err:
                entry["status"] = str(err)
                continue
            entry.update(timing)
            lo = len(matrix)
            matrix.append([*core, lower])
            hi = lo
            if upper != lower:
                hi = len(matrix)
                matrix.append([*core, upper])
            scores.append((entry, lo, hi, fraction))

        if matrix:
            values = np.asarray(matrix, dtype=np.float32)
            raw = self.model.predict(values, raw_score=True, num_threads=2)
            current = values[:, self.input_names.index("availableCount")] >= 1
            p = probabilities(raw, values[:, -1], current, self.card["calibrators"])
            for entry, lo, hi, fraction in scores:
                entry["probability"] = float(p[lo] + (p[hi] - p[lo]) * fraction)
        return {
            "requestId": request.get("requestId"), "modelVersion": self.card["model_version"],
            "featureVersion": self.card["feature_version"],
            "items": output,
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    print(json.dumps(ETAPredictorV9(args.model_dir).predict_batch(request), ensure_ascii=False, indent=2))
