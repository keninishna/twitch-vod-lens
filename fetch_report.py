"""Use pexpect.run to cat JSON from WSL2 and parse it."""
import pexpect, json

out = pexpect.run(
    'ssh -o StrictHostKeyChecking=no john@100.97.240.34 cat /home/john/twitch-vod-analyzer/vods/phase4_2770929139/qwen_vision_progressive.json',
    events={'password:': 'Sparky1234\n'},
    timeout=30, encoding='utf-8'
)

print("OUTPUT LENGTH:", len(out))
# Strip SSH password prompt noise
idx = out.find('{')
if idx >= 0:
    out = out[idx:]
d = json.loads(out)

for c in d['final_ranking']['final_selected_clips']:
    s = c['start']
    sc = c['score']
    cp = c.get('clip_point', '')
    ir = c.get('intelligence_report', {})
    ps = c.get('platform_scores', {})
    print(f"\nClip {s}s (score={sc})")
    print(f"  Title: {cp}")
    print(f"  Why: {ir.get('why_selected', '')}")
    print(f"  Arc: {ir.get('narrative_arc', '')}")
    print(f"  Evidence: {ir.get('evidence', '')}")
    print(f"  Trim: {ir.get('trim_rationale', '')[:200]}")
    print(f"  Duration: {ir.get('duration_fit', '')}")
    print(f"  Platform: {ir.get('platform_fit', '')}")
    print(f"  Risks: {ir.get('risks', '')}")
    print(f"  Scores: {dict(ps)}")
    print(f"  Recs: {c.get('platform_recommendations', [])}")
