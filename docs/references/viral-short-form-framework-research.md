# Viral Short-Form Framework: HPC (Hook-Progression-Climax)

> **Source:** Logan E. Smith (@LoganEdwardSmith), *"How to make shorts that go viral every time"*
> https://youtu.be/gEQ0BLyVJhY
> **Extracted Date:** May 18, 2026
> **Purpose:** Apply short-form video virality science to clip selection criteria for the VOD Lens Intelligence pipeline.

---

## Core Framework: HPC System

The video's central framework for viral short-form content is the **HPC System** — Hook, Progression, Climax. These three elements determine viewer retention, which is the single most important signal YouTube's algorithm uses to push short-form content.

### 1. Hook (First ~5 Seconds)

| Principle | Description |
|-----------|-------------|
| **Introduce topic + leave curiosity** | The hook must both announce what the video is about AND leave the viewer wondering what happens next |
| **No wasted time** | Every millisecond counts; cutting dead air, slow intros, or establishing shots that don't serve the hook |
| **Pattern interrupt** | The hook should disrupt the viewer's scrolling pattern — surprising statement, unusual visual, question, or bold claim |

**Clip selection relevance:**
- When evaluating clips, assess whether the **first 5 seconds** of the clip contain a clear hook
- Clips that start with dead air, rambling, or slow buildup should be penalized (or trimmed to start at the hook moment)
- A clip without a clear hook is unlikely to perform as a standalone short, even if the middle is good

### 2. Progression

| Principle | Description |
|-----------|-------------|
| **Fulfill hook expectations** | The body of the short delivers on what the hook promised — a journey aligned with the initial setup |
| **Pacing** | Information should be revealed progressively, not all at once. Key reveals should be delayed to maintain retention |
| **No "front-loading"** | Showing the best moment too early kills retention. The viewer has no reason to keep watching after seeing the payoff |
| **Structure = storytelling** | Even in 30 seconds, there should be a mini-arc: setup → tension/development → resolution |

**Clip selection relevance:**
- Clips where the best moment happens in the **first 10 seconds** are poor candidates (no progression)
- Evaluate narrative pacing: does the clip build toward something, or is it flat throughout?
- Trim narrow windows should preserve the setup-payoff relationship, not just the payoff
- A clip that is "all hook, no progression" (constant high energy, no build) often has lower retention

### 3. Climax

| Principle | Description |
|-----------|-------------|
| **The payoff** | The moment that justifies watching — the laugh, reveal, twist, or emotional hit |
| **Proportional to setup** | The climax should feel earned relative to the progression that led there |
| **Standalone clarity** | The climax must be understandable without context from outside the clip |

**Clip selection relevance:**
- The climax is **not** the same as "the funniest 3-second burst seconds." A climax that works standalone may still fail if the progression to it is missing
- Clips that cut straight to a climax with no setup are low-retention candidates (viewers are confused)
- The highest-scoring clips have clear climax moments that feel proportional to the setup

---

## Algorithm & Retention Principles

### The Algorithm is a Retention Filter

> "The algorithm shows the best, most engaging content. It's not random or based on luck."

- YouTube's algorithm surfaces content with **high retention** (viewed≥ swiped-away ratio
- Retention is determined by **HPC structure** — not production value, not luck
- Creators who consistently go viral understand the science of retention, not "hacks"

**Clip selection relevance:**
- Score clips not just on what happens, but on **retention trajectory** — does the clip keep a viewer watching?
- A clip with high emotional energy but no structure (no hook, no build) is unlikely to retain viewers as a standalone short
- Long clips (>60s) need proportionally stronger HPC structure to justify the duration

### Topic Selection Matters More Than Execution

> "An interesting topic is crucial for virality. Research successful channels in your niche."

- Even perfect HPC structure won't save a boring topic
- The most viral clips come from topics the intersection of: **interesting topic × strong structure × good editing**
- Audience targeting: a clip that is perfectly structured for the wrong audience will underperform

**Clip selection relevance:**
- When a clip covers a niche/inside-joke topic, it should be scored lower for broad platform fitness (TikTok, Shorts, Reels) but may still score well for Twitch
- Clip topics should be evaluated for **universal appeal** vs **niche appeal** — both are valid but have different platform recommendations
- A clip about an inside joke needs enough context built in to be standalone-hooky

---

## Posting & Timing Factors (Pipeline Context)

While these apply to upload strategy, they have indirect relevance to clip selection:

| Factor | Insight | Relevance to Selection |
|--------|---------|----------------------|
| **Posting time** | Morning posts received higher views based on audience analytics | Not directly applicable, but: clips that depend on "live" chat interaction may lose relevance over time |
| **Frequency** | 4 shorts/day was excessive and lowered quality | Relevant for the extraction batch: prefer 2-3 high-quality clips over 8 mediocre ones |

---

## Example: HPC Applied to a Real Clip

From the video itself (snippet from transcript found on pickscribe.com):
> "HPC progression but there's a problem it's hot out here the sand won't form balls..."

The video demonstrates HPC through a **snowball fight in the desert** example:
- **Hook:** Desert snowball fight? That's unexpected — curiosity created
- **Progression:** The attempt to make snowballs in sand, the failures, the problem
- **Climax:** The resolution/punchline of the snowball fight bit

This shows that HPC works even for seemingly trivial or silly content — the structure, not the subject, drives retention.

---

## Application to Lens Pipeline Scoring

### Suggested Prompt Additions for Stage 1 & Stage 3

Add these criteria to clip evaluation prompts:

```
### HPC (Hook-Progression-Climax) Evaluation

Rate the clip on these three dimensions:

1. **HOOK (0-10):** Does the clip start with a clear hook in the first 5 seconds?
   - 10 = Immediate pattern interrupt + curiosity gap
   - 5 = Decent opener but no real hook
   - 0 = Dead air, rambling, or slow start

2. **PROGRESSION (0-10):** Does the clip build toward something?
   - 10 = Clear mini-arc with delayed payoff
   - 5 = Flat energy throughout, no build
   - 0 = Best moment happens in first 10 seconds

3. **CLIMAX (0-10):** Is there a satisfying payoff?
   - 10 = Clear climax that feels earned and is standalone-comprehensible
   - 5 = Decent moment but context-dependent
   - 0 = No climax, or climax requires outside context

**HPC Score = average of all three**

Clips with HPC Score < 5 should not score >= 8 overall even if the content is "funny"
— a funny moment without structure will not retain viewers as a standalone short.
```

### Suggested Stage 3 Penalty/Adjustment

Add a deterministic **HPC penalty** in Stage 2 scoring (in `scoring.py`):

```python
def hpc_penalty(hook_score: float, progression_score: float,
                climax_score: float) -> float:
    """Return a score penalty based on HPC structure weakness.

    Weakness definitions:
    - hook < 4: -1.5 (no entry point for viewer)
    - progression < 3: -2.0 (no build — viewer leaves early)
    - climax < 4: -1.0 (weak payoff — viewer unsatisfied)
    - best_moment_too_early: -2.0 (front-loaded, no reason to finish)
    """
    penalty = 0.0
    if hook_score < 4: penalty += 1.5
    if progression_score < 3: penalty += 2.0
    if climax_score < 4: penalty += 1.0
    return penalty
```

The HPC framework **replaces** qualitative judgments like "emotional energy" or "funnyness" with a structured, retention-based model grounded in short-form video science.

---

## Comparison to Existing Criteria

| Existing Criterion | HPC Equivalence | Notes |
|--------------------|----------------|-------|
| Narrative arc | Progression + Climax | HPC splits this into two measurable dimensions |
| Transactional reaction detection | Low Progression score | A reaction without build is flat progression |
| Standalone clarity | Weak Hook score | If context is needed, the hook failed |
| Dead air detection | Low Hook score | Dead air in first 5s kills hook entirely |
| Emotional energy | Not HPC | Energy without structure scores low on HPC |

A clip can have high emotional energy and still score poorly on HPC — which explains why "laughing at donation" clips struggle to go viral as shorts.

---

## References

- Smith, Logan E. *"How to make shorts that go viral every time."* YouTube, @LoganEdwardSmith.
  https://youtu.be/gEQ0BLyVJhY
- VideoHighlight compressed summary (gEQ0BLyVJhY)
- Scribe/pickscribe.com transcript snapshot (gEQ0BLyVJhY)
- Applied to short-form clip analysis via Lens Intelligence Pipeline