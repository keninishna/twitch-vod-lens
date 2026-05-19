"""Fetch pipeline_v2 results with merge."""
import pexpect, json

out = pexpect.run(
    'ssh -o StrictHostKeyChecking=no john@100.97.240.34 cat /home/john/twitch-vod-analyzer/vods/phase4_2770929139/qwen_vision_progressive.json',
    events={'password:': 'Sparky1234\n'},
    timeout=20, encoding='utf-8'
)
idx = out.find('{')
data = json.loads(out[idx:])

# Check log header
print("=== LOG HEADER ===")
out2 = pexpect.run(
    'ssh -o StrictHostKeyChecking=no john@100.97.240.34 grep -E \"Stage 2 scoring|Final selection|passed score\" /tmp/pipeline_v2.log',
    events={'password:': 'Sparky1234\n'},
    timeout=15, encoding='utf-8'
)
idx2 = out2.find('Stage')
if idx2 >= 0:
    print(out2[idx2:])

for c in data['final_ranking']['final_selected_clips']:
    s = c['start']
    sc = c['score']
    cp = c.get('clip_point','')
    ir = c.get('intelligence_report',{})
    ps = c.get('platform_scores',{})
    print(f'\n=== Clip {s}s (score={sc}) ===')
    print(f'  Title: {cp}')
    print(f'  Why: {ir.get("why_selected","")[:120]}')
    print(f'  Strengths: {ir.get("strengths",[])}')
    print(f'  Platform scores: {json.dumps(ps)}')
    print(f'  Recs: {c.get("platform_recommendations",[])}')
    print(f'  Risks: {ir.get("risks",[])}')
