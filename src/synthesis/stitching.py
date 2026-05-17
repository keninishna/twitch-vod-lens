"""Stage 1.5 deterministic cross-window stitching.

This module merges Stage 1 discovery candidates into stitched story arcs while
preserving provenance fields required by the stage contract.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Set, Tuple

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


def _candidate_id(candidate: Dict) -> str:
    return str(candidate.get("candidate_id") or f"cand_{_as_int(candidate.get('start'), 0)}")


def _candidate_tokens(candidate: Dict) -> Set[str]:
    chunks: List[str] = [
        str(candidate.get("trigger") or ""),
        str(candidate.get("payoff") or ""),
        str(candidate.get("narrative_type") or ""),
    ]
    chunks.extend(str(x) for x in (candidate.get("evidence_lines") or []))
    return _tokenize(" ".join(chunks))


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _score_pair_merge(
    left: Dict,
    right: Dict,
    max_gap_seconds: int,
    min_shared_tokens: int,
    max_bridge_gap_seconds: int,
) -> Tuple[bool, List[str], Dict]:
    """Score whether two candidates should belong to the same stitched component.

    Returns:
      - should_merge
      - human-readable reason codes
      - debug record payload
    """

    left_end = _as_int(left.get("end"), 0)
    right_start = _as_int(right.get("start"), 0)
    gap = right_start - left_end

    left_type = str(left.get("narrative_type") or "unknown")
    right_type = str(right.get("narrative_type") or "unknown")
    type_match = left_type == right_type

    shared_tokens = sorted(_candidate_tokens(left).intersection(_candidate_tokens(right)))
    shared_count = len(shared_tokens)

    score = 0
    reasons: List[str] = []

    if gap <= max_gap_seconds:
        score += 2
        reasons.append(f"temporal_gap<={max_gap_seconds}")
    elif gap <= max_bridge_gap_seconds:
        score += 1
        reasons.append(f"temporal_bridge_gap<={max_bridge_gap_seconds}")

    if type_match:
        score += 2
        reasons.append(f"narrative_type_match:{left_type}")

    if shared_count >= min_shared_tokens:
        score += 2
        reasons.append("shared_tokens:" + ",".join(shared_tokens[:6]))
    elif shared_count > 0:
        score += 1
        reasons.append("weak_shared_tokens:" + ",".join(shared_tokens[:4]))

    # Backward-compatible strict local merge rule + broader graph rule.
    strict_local_rule = gap <= max_gap_seconds and (type_match or shared_count >= min_shared_tokens)
    graph_rule = gap <= max_bridge_gap_seconds and score >= 4
    should_merge = strict_local_rule or graph_rule

    debug_record = {
        "left_candidate_id": _candidate_id(left),
        "right_candidate_id": _candidate_id(right),
        "left_window": [_as_int(left.get("start"), 0), _as_int(left.get("end"), 0)],
        "right_window": [_as_int(right.get("start"), 0), _as_int(right.get("end"), 0)],
        "gap_seconds": gap,
        "type_match": type_match,
        "shared_token_count": shared_count,
        "shared_tokens": shared_tokens[:8],
        "score": score,
        "strict_local_rule": strict_local_rule,
        "graph_rule": graph_rule,
        "merged": should_merge,
        "reasons": reasons,
    }

    return should_merge, reasons, debug_record


def _aggregate_cluster(cluster: List[Dict], stitched_idx: int, merge_reasons: Optional[List[str]] = None) -> Dict:
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

    reason_list = _dedupe_preserve_order(merge_reasons or [])
    if not reason_list:
        reason_list = ["single_candidate"]

    stitched = {
        "stitched_id": f"stitched_{min(starts)}_{max(ends)}_{stitched_idx}",
        "start": min(starts),
        "end": max(ends),
        "narrative_type": narrative_type,
        "trigger": trigger,
        "payoff": payoff,
        "evidence_lines": evidence,
        "confidence": round(max(_as_float(c.get("confidence"), 0.0) for c in cluster), 4),
        "source_candidate_ids": [_candidate_id(c) for c in cluster],
        "source_windows": [[_as_int(c.get("start"), 0), _as_int(c.get("end"), 0)] for c in cluster],
        "merge_reasons": reason_list,
    }

    validated = validate_stage_payload("stitched", stitched)
    return validated.model_dump()


def stitch_discoveries(
    discoveries: List[Dict],
    max_gap_seconds: int = 20,
    min_shared_tokens: int = 2,
    max_bridge_gap_seconds: int = 45,
    max_cluster_span_seconds: int = 240,
    debug_decisions: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Deterministically stitch discoveries with pairwise graph merging.

    Compared to adjacency-only merging, this allows story arcs to stitch across
    noisy/misaligned intermediate candidates when evidence supports continuity.

    Edge rule:
    - pair gap <= max_bridge_gap_seconds, and
    - weighted evidence score >= threshold (or strict local adjacency rule).

    Component guard:
    - merged component span cannot exceed max_cluster_span_seconds.
    """

    ordered = sorted(discoveries, key=lambda d: (_as_int(d.get("start"), 0), _as_int(d.get("end"), 0)))
    if not ordered:
        return []

    n = len(ordered)

    # Build candidate merge edges.
    edge_candidates: List[Tuple[int, int, List[str], int, int]] = []
    for i in range(n):
        left = ordered[i]
        left_end = _as_int(left.get("end"), 0)
        for j in range(i + 1, n):
            right = ordered[j]
            gap = _as_int(right.get("start"), 0) - left_end
            if gap > max_bridge_gap_seconds:
                break

            should_merge, reasons, debug_record = _score_pair_merge(
                left=left,
                right=right,
                max_gap_seconds=max_gap_seconds,
                min_shared_tokens=min_shared_tokens,
                max_bridge_gap_seconds=max_bridge_gap_seconds,
            )

            if debug_decisions is not None:
                debug_decisions.append(debug_record)

            if should_merge:
                edge_candidates.append((i, j, reasons, debug_record["score"], gap))

    # DSU helpers.
    parent = list(range(n))
    rank = [0] * n
    root_min_start = [_as_int(c.get("start"), 0) for c in ordered]
    root_max_end = [_as_int(c.get("end"), 0) for c in ordered]

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    accepted_edges: List[Tuple[int, int, List[str]]] = []

    # Prefer stronger edges first, then tighter temporal gaps.
    edge_candidates.sort(key=lambda e: (-e[3], e[4], e[0], e[1]))

    for i, j, reasons, score, gap in edge_candidates:
        ri = find(i)
        rj = find(j)
        if ri == rj:
            continue

        merged_min = min(root_min_start[ri], root_min_start[rj])
        merged_max = max(root_max_end[ri], root_max_end[rj])
        merged_span = merged_max - merged_min

        if merged_span > max_cluster_span_seconds:
            if debug_decisions is not None:
                debug_decisions.append({
                    "left_candidate_id": _candidate_id(ordered[i]),
                    "right_candidate_id": _candidate_id(ordered[j]),
                    "gap_seconds": gap,
                    "score": score,
                    "merged": False,
                    "reasons": ["span_guard_exceeded"],
                    "component_span_seconds": merged_span,
                    "max_cluster_span_seconds": max_cluster_span_seconds,
                })
            continue

        if rank[ri] < rank[rj]:
            ri, rj = rj, ri
        parent[rj] = ri
        if rank[ri] == rank[rj]:
            rank[ri] += 1

        root_min_start[ri] = merged_min
        root_max_end[ri] = merged_max
        accepted_edges.append((i, j, reasons))

    # Collect connected components.
    groups: Dict[int, List[int]] = {}
    for idx in range(n):
        root = find(idx)
        groups.setdefault(root, []).append(idx)

    stitched: List[Dict] = []
    for stitched_idx, indices in enumerate(sorted(groups.values(), key=lambda g: min(_as_int(ordered[i].get("start"), 0) for i in g)), start=1):
        sorted_indices = sorted(indices, key=lambda i: (_as_int(ordered[i].get("start"), 0), _as_int(ordered[i].get("end"), 0)))
        cluster = [ordered[i] for i in sorted_indices]

        if len(cluster) == 1:
            group_reasons = ["single_candidate"]
        else:
            idx_set = set(sorted_indices)
            group_reasons = _dedupe_preserve_order(
                reason
                for i, j, reasons in accepted_edges
                if i in idx_set and j in idx_set
                for reason in reasons
            )
            if not group_reasons:
                group_reasons = ["graph_component_merge"]

        stitched.append(_aggregate_cluster(cluster, stitched_idx=stitched_idx, merge_reasons=group_reasons))

    return stitched
