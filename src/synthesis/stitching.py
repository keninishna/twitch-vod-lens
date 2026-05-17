"""Stage 1.5 deterministic cross-window stitching.

This module merges adjacent Stage 1 discovery candidates into stitched story arcs
while preserving provenance fields required by the stage contract.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Set

from src.synthesis.schemas import validate_stage_payload

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
    "is", "it", "this", "that", "she", "he", "they", "you", "chat", "streamer",
    "from", "at", "as", "be", "was", "are", "were", "what", "when", "why",
}


def _as_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value, default=0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _tokenize(text: str) -> Set[str]:
    toks = {
        t
        for t in _TOKEN_RE.findall((text or "").lower())
        if len(t) >= 3 and t not in _STOPWORDS
    }
    return toks


def _candidate_tokens(candidate: Dict) -> Set[str]:
    chunks: List[str] = [
        str(candidate.get("trigger") or ""),
        str(candidate.get("payoff") or ""),
        str(candidate.get("narrative_type") or ""),
    ]
    chunks.extend(str(x) for x in (candidate.get("evidence_lines") or []))
    return _tokenize(" ".join(chunks))


def _should_merge(left: Dict, right: Dict, max_gap_seconds: int, min_shared_tokens: int):
    left_end = _as_int(left.get("end"), 0)
    right_start = _as_int(right.get("start"), 0)
    gap = right_start - left_end

    if gap > max_gap_seconds:
        return False, []

    reasons: List[str] = [f"temporal_gap<={max_gap_seconds}"]

    left_type = str(left.get("narrative_type") or "unknown")
    right_type = str(right.get("narrative_type") or "unknown")
    if left_type == right_type:
        reasons.append(f"narrative_type_match:{left_type}")
        return True, reasons

    shared = sorted(_candidate_tokens(left).intersection(_candidate_tokens(right)))
    if len(shared) >= min_shared_tokens:
        reasons.append("shared_tokens:" + ",".join(shared[:6]))
        return True, reasons

    return False, []


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _aggregate_cluster(cluster: List[Dict], stitched_idx: int) -> Dict:
    starts = [_as_int(c.get("start"), 0) for c in cluster]
    ends = [_as_int(c.get("end"), 0) for c in cluster]

    narrative_counter = Counter(str(c.get("narrative_type") or "unknown") for c in cluster)
    narrative_type = narrative_counter.most_common(1)[0][0]

    trigger = " | ".join(
        _dedupe_preserve_order(str(c.get("trigger") or "") for c in cluster if c.get("trigger"))
    ) or "Model did not provide explicit trigger"
    payoff = " | ".join(
        _dedupe_preserve_order(str(c.get("payoff") or "") for c in cluster if c.get("payoff"))
    ) or "Model did not provide explicit payoff"

    evidence: List[str] = []
    for c in cluster:
        for line in (c.get("evidence_lines") or []):
            line_str = str(line).strip()
            if line_str and line_str not in evidence:
                evidence.append(line_str)
    if not evidence:
        evidence = ["No direct evidence provided by model output"]

    stitched = {
        "stitched_id": f"stitched_{min(starts)}_{max(ends)}_{stitched_idx}",
        "start": min(starts),
        "end": max(ends),
        "narrative_type": narrative_type,
        "trigger": trigger,
        "payoff": payoff,
        "evidence_lines": evidence,
        "confidence": round(max(_as_float(c.get("confidence"), 0.0) for c in cluster), 4),
        "source_candidate_ids": [str(c.get("candidate_id") or f"cand_{_as_int(c.get('start'), 0)}") for c in cluster],
        "source_windows": [[_as_int(c.get("start"), 0), _as_int(c.get("end"), 0)] for c in cluster],
        "merge_reasons": _dedupe_preserve_order(
            reason
            for c in cluster
            for reason in (c.get("_merge_reasons") or [])
            if reason
        ) or ["single_candidate"],
    }

    validated = validate_stage_payload("stitched", stitched)
    return validated.model_dump()


def stitch_discoveries(
    discoveries: List[Dict],
    max_gap_seconds: int = 20,
    min_shared_tokens: int = 2,
) -> List[Dict]:
    """Deterministically merge adjacent Stage 1 discoveries.

    Two candidates merge when:
    - temporal gap <= max_gap_seconds, and
    - either narrative_type matches OR they share enough narrative tokens.
    """

    ordered = sorted(discoveries, key=lambda d: (_as_int(d.get("start"), 0), _as_int(d.get("end"), 0)))
    if not ordered:
        return []

    stitched_clusters: List[List[Dict]] = []
    current_cluster: List[Dict] = [dict(ordered[0], _merge_reasons=["single_candidate"])]

    for nxt in ordered[1:]:
        prev = current_cluster[-1]
        should_merge, reasons = _should_merge(
            left=prev,
            right=nxt,
            max_gap_seconds=max_gap_seconds,
            min_shared_tokens=min_shared_tokens,
        )

        nxt_copy = dict(nxt)
        if should_merge:
            # Replace default singleton marker with real merge reasons.
            if current_cluster[-1].get("_merge_reasons") == ["single_candidate"]:
                current_cluster[-1]["_merge_reasons"] = []
            nxt_copy["_merge_reasons"] = reasons
            current_cluster.append(nxt_copy)
        else:
            stitched_clusters.append(current_cluster)
            nxt_copy["_merge_reasons"] = ["single_candidate"]
            current_cluster = [nxt_copy]

    stitched_clusters.append(current_cluster)

    stitched: List[Dict] = []
    for idx, cluster in enumerate(stitched_clusters, start=1):
        stitched.append(_aggregate_cluster(cluster, stitched_idx=idx))

    return stitched
