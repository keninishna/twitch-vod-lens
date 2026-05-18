"""Fetch final merged results."""
import pexpect, json

out = pexpect.run(
    'ssh -o StrictHostKeyChecking=no john@100.97.240.34 cat /home/john/twitch-vod-analyzer/vods/phase4_2770929139/qwen_vision_progressive.json',
    events={'password:': 'Sparky1234\n'},
    timeout=20, encoding='utf-8'
)
idx = out.find('{')
if idx < 0:
    print('NO JSON:', out[:300])
else:
    data = json.loads(out[idx:])
    for c in data['final_ranking']['final_selected_clips']:
        s = c['start']
        sc = c['score']
        cp = c.get('clip_point','')
        ir = c.get('intelligence_report',{})
        ps = c.get('platform_scores',{})
        print(f'=== Clip {s}s (score={sc}) ===')
        print(f'  Title: {cp}')
        print(f'  Why: {ir.get("why_selected","")}')
        print(f'  Strengths: {ir.get("strengths",[])}')
        print(f'  Weaknesses: {ir.get("weaknesses",[])}')
        print(f'  Narrative quality: {ir.get("narrative_quality","")}')
        print(f'  Platform scores: {json.dumps(ps)}')
        print(f'  Platform recs: {c.get("platform_recommendations",[])}')
        print()