"""Fetch v5 full intelligence for selected clip + model picks."""
import pexpect, json

out = pexpect.run(
    'ssh -o StrictHostKeyChecking=no john@100.97.240.34 cat /home/john/twitch-vod-analyzer/vods/phase4_2770929139/qwen_vision_progressive.json',
    events={'password:': 'Sparky1234\n'},
    timeout=20, encoding='utf-8'
)
idx = out.find('{')
data = json.loads(out[idx:])

print("=== DETERMINISTIC CLIP DETAILS (Stage 1 raw scores) ===")
for c in data.get("clip_details", []):
    a = c.get("analysis", {})
    s = c.get("start", "?")
    cw = a.get("clip_worthiness", "?")
    nt = a.get("narrative_type", "?")
    cp = str(a.get("clip_point", ""))[:80]
    print(f"  {s}s: raw_score={cw} type={nt}")
    print(f"    clip_point: {cp}")

print("\n=== QWEN's FINAL SYNTHESIS (model_final_selected_clips) ===")
for c in data['final_ranking'].get('model_final_selected_clips', []):
    print(json.dumps(c, indent=2))
    print()

print("=== MERGED FINAL SELECTED CLIP ===")
c = data['final_ranking']['final_selected_clips'][0]
ir = c.get('intelligence_report', {})
print(json.dumps({
    'start': c['start'],
    'score': c['score'],
    'clip_point': c.get('clip_point'),
    'why_selected': ir.get('why_selected'),
    'platform_scores': c.get('platform_scores'),
    'recommendations': c.get('platform_recommendations'),
    'strengths': ir.get('strengths'),
    'weaknesses': ir.get('weaknesses'),
    'narrative_quality': ir.get('narrative_quality'),
    'platform_reasoning': c.get('platform_reasoning'),
}, indent=2))