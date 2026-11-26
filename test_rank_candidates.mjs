import { deepEqual, equal } from "node:assert/strict";
import { rankCandidates } from "./rank_candidates.mjs";

const candidates = [
  { stationId: "uncertain", distanceKm: .7, probability: null },
  { stationId: "far", distanceKm: 4, probability: .95 },
  { stationId: "near", distanceKm: 1, probability: .68 },
  { stationId: "middle", distanceKm: 2, probability: .90 },
];
const ids = (items) => items.map(({ stationId }) => stationId);
deepEqual(ids(rankCandidates(candidates, "distance")), ["uncertain", "near", "middle", "far"]);
deepEqual(ids(rankCandidates(candidates, "probability")), ["far", "middle", "near", "uncertain"]);
deepEqual(ids(rankCandidates(candidates, "balanced")), ["middle", "near", "far", "uncertain"]);
equal(rankCandidates(candidates)[0].balancedScore.toFixed(2), "5.50");
equal(rankCandidates(candidates).at(-1).balancedScore, null);
deepEqual(ids(candidates), ["uncertain", "far", "near", "middle"]);
console.log("All recommendation order and missing-probability checks passed.");
