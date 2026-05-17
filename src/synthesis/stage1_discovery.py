"""Stage 1 discovery-only helpers.

These helpers normalize per-clip model outputs into discovery payloads that are
safe for Stage 1 carryover (no title generation, no platform-finalization
fields).
"""

from __future__ import annotations

from typing import Dict, List

from src.synthesis.schemas import validate_stage_payload


def _as_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _confidence_from_analysis(analysis: Dict) -> float:
    # Prefer explicit confidence if present.
    explicit = analysis.get("confidence")
    if explicit is not None:
        c = _as_float(explicit, 0.0)
        return min(1.0, max(0.0, c))

    # Fallback from clip_worthiness (1-10) -> 0.1-1.0.
    score = _as_float(analysis.get("clip_worthiness"), 0.0)
    if score <= 0:
        return 0.3
    return min(1.0, max(0.0, score / 10.0))


def map_analysis_to_discovery(clip: Dict, analysis: Dict) -> Dict:
    """Return a discovery-only payload from an analysis object."""

    start = int(_as_float(clip.get("start"), 0))
    end = int(_as_float(clip.get("end"), start))

    narrative_type = str(analysis.get("narrative_type") or "unknown")
    narrative_arc = str(analysis.get("narrative_arc") or "")
    reason = str(analysis.get("reason") or "")

    trigger = str(analysis.get("trigger") or "Model did not provide explicit trigger")
    payoff = str(analysis.get("payoff") or analysis.get("has_narrative_payoff") or "Model did not provide explicit payoff")

    evidence_lines: List[str] = []
    if reason:
        evidence_lines.append(reason)
    if narrative_arc:
        evidence_lines.append(f"narrative_arc: {narrative_arc}")

    title = str(analysis.get("clip_point") or "").strip()
    if title:
        evidence_lines.append(f"clip_point: {title}")

    if not evidence_lines:
        evidence_lines.append("No direct evidence provided by model output")

    discovery_payload = {
        "candidate_id": f"cand_{start}",
        "start": start,
        "end": end,
        "narrative_type": narrative_type,
        "trigger": trigger,
        "payoff": str(payoff),
        "evidence_lines": evidence_lines,
        "confidence": round(_confidence_from_analysis(analysis), 4),
    }

    validated = validate_stage_payload("discovery", discovery_payload)
    return validated.model_dump()


def build_discovery_batch_context(all_results: List[Dict], total: int, batch_idx: int) -> str:
    """Build discovery-only running context for Stage 1 batch carryover."""

    lines = [
        f"Analysed {len(all_results)}/{total} clips so far (through batch {batch_idx}).",
        "Top discoveries so far:",
    ]

    with_discovery = [r for r in all_results if isinstance(r.get("discovery"), dict)]
    ranked = sorted(
        with_discovery,
        key=lambda r: _as_float(r.get("discovery", {}).get("confidence"), 0.0),
        reverse=True,
    )

    for r in ranked[:5]:
        d = r["discovery"]
        lines.append(
            "  - "
            f"{r.get('start','?')}s: narrative={d.get('narrative_type','?')}, "
            f"confidence={_as_float(d.get('confidence'), 0.0):.2f}, "
            f"trigger={str(d.get('trigger',''))[:50]}, "
            f"payoff={str(d.get('payoff',''))[:50]}"
        )

    high_conf = sum(1 for r in with_discovery if _as_float(r["discovery"].get("confidence"), 0.0) >= 0.7)
    lines.append(f"\nCross-batch observations so far: {high_conf} discoveries with confidence >= 0.7.")

    return "\n".join(lines)
