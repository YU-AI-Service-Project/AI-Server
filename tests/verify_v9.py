"""Check chronological feature reconstruction and prediction-only HTTP behavior."""

from copy import deepcopy
import argparse
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import os
import sys
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))
from api_server import app  # noqa: E402
from event_features import build  # noqa: E402
RESULTS = ROOT / "results"
ENDPOINT = "/internal/v3/arrival-predictions/batch"


def read(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def digest(name):
    with (RESULTS / name).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def verify_causality(source):
    cutoff = "2026-11-14T12:00"
    with TemporaryDirectory(prefix="wattup-v9-events-") as temp:
        prefix = Path(temp)
        build(source, ROOT / "training" / "state_replay", prefix, cutoff)
        counts = {}
        for name in ("ALL", "AC_SLOW", "DC_COMBO"):
            a = np.load(prefix / f"{name}_events.npy", mmap_mode="r")
            b = np.load(ROOT / "training" / "event_data" / f"{name}_events.npy", mmap_mode="r")
            try:
                for start in range(0, len(a), 48):
                    end = min(start + 48, len(a))
                    np.testing.assert_array_equal(a[start:end], b[start:end])
                counts[name] = int(a.shape[0] * a.shape[1])
            finally:
                del a, b
    return {"causal_cutoff_KST": cutoff, "historical_feature_rows_unchanged": counts}


def verify_api():
    card = read("model_card.json")
    lock = read("selection_lock.json")
    assert digest("model_card.json") == lock["card_sha256"]
    assert digest(card["model_file"]) == lock["model_sha256"]
    cases = read("verification_fixtures.json")
    count = 0
    maximum = 0.
    with TestClient(app) as client:
        assert client.get("/health/ready").json()["modelVersion"] == card["model_version"]
        secret=os.environ.pop("WATTUP_INTERNAL_API_KEY", None)
        try:
            os.environ["WATTUP_INTERNAL_API_KEY"]="local-unit-test-only"
            sample=cases[0]["request"]
            assert client.post(ENDPOINT,json=sample).status_code == 401
            assert client.post(ENDPOINT,json=sample,headers={"Authorization":"Bearer wrong"}).status_code == 401
            assert client.post(ENDPOINT,json=sample,headers={"Authorization":"Bearer local-unit-test-only"}).status_code == 200
            assert client.post(ENDPOINT,json=sample,headers={"X-Internal-Api-Key":"local-unit-test-only"}).status_code == 200
        finally:
            os.environ.pop("WATTUP_INTERNAL_API_KEY",None)
        if secret:
            client.headers["Authorization"]="Bearer "+secret
            os.environ["WATTUP_INTERNAL_API_KEY"]=secret
        for case in cases:
            request = case["request"]
            response = client.post(ENDPOINT, json=request)
            response.raise_for_status()
            items = response.json()["items"]
            assert [i["requestItemId"] for i in items] == [i["requestItemId"] for i in request["items"]]
            assert all(i["status"] == "OK" for i in items)
            assert all("arrivalRank" not in i and "recommendation" not in i for i in items)
            maximum = max(maximum, float(np.max(np.abs(
                np.array([i["probability"] for i in items]) - case["expected_probabilities"],
            ))))
            count += len(items)
        assert maximum < 1e-10

        request = deepcopy(cases[0]["request"])
        sample = request["items"][0]
        expired = datetime.fromisoformat(sample["asOf"]) - timedelta(minutes=8)
        incomplete = deepcopy(sample)
        incomplete["requestItemId"] = "missing-history"
        incomplete["features"].pop("historyKnown30m")
        request["items"] = [sample, dict(sample, requestItemId="expired", asOf=expired.isoformat()), incomplete]
        rows = client.post(ENDPOINT, json=request).json()["items"]
        assert [i["status"] for i in rows] == ["OK", "STALE_FEATURES", "INPUT_INCOMPLETE"]
        assert rows[1]["probability"] is None and rows[2]["probability"] is None
        request["items"] = [dict(sample, requestItemId=str(i)) for i in range(10)]
        response = client.post(ENDPOINT, json=request)
        assert response.status_code == 200 and len(response.json()["items"]) == 10
        request["items"].append(dict(sample, requestItemId="11"))
        assert client.post(ENDPOINT, json=request).status_code == 422
    return {
        "http_replayed_batches": len(cases), "http_replayed_items": count, "max_probability_error": maximum,
        "batch_10_ok_11_rejected": True, "partial_unavailable_verified": True,
        "spring_boot_bearer_auth_verified": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Optionally verify causal training features using the original ZIP")
    args = parser.parse_args()
    checked = verify_api()
    if args.source is not None:
        checked.update(verify_causality(args.source))
    print(json.dumps(checked, indent=2))
