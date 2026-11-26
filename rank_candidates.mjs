/**
 * Rank one loaded list. Distance is in km; probability describes availability
 * at the candidate's own predicted arrival and ranges from 0 to 1.
 */
export function rankCandidates(candidates, mode = "balanced") {
  if (!["balanced", "distance", "probability"].includes(mode)) {
    throw new RangeError("Unknown candidate sort");
  }
  return candidates.map((item) => {
    const p = item.probability;
    const km = item.distanceKm;
    const knownProbability = Number.isFinite(p) && p >= 0 && p <= 1;
    const knownDistance = Number.isFinite(km) && km >= 0;
    const balanced = knownProbability && knownDistance ? 2 * km + 15 * (1 - p) : null;
    return {
      item: { ...item, balancedScore: balanced },
      byDistance: knownDistance ? km : Infinity,
      byProbability: knownProbability ? -p : Infinity,
      byBalanced: balanced ?? Infinity,
    };
  }).sort((a, b) => {
    const { first, second } = {
      distance: { first: "byDistance", second: "byProbability" },
      probability: { first: "byProbability", second: "byDistance" },
      balanced: { first: "byBalanced", second: "byDistance" },
    }[mode];
    return (a[first] - b[first]) || (a[second] - b[second]);
  }).map(({ item }) => item);
}
