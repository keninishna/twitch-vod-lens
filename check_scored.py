"""Check all scored candidates and their final scores."""
import pexpect, json

out = pexpect.run(
    'ssh -o StrictHostKeyChecking=no john@100.97.240.34 cat /home/john/twitch-vod-analyzer/vods/phase4_2770929139/qwen_vision_progressive.json',
    events={'password:': 'Sparky1234\n'},
    timeout=20, encoding='utf-8'
)
idx = out.find('{')
data = json.loads(out[idx:])

print("=== ALL SCORED CANDIDATES ===")
for s in data.get('stage2_scored', []):
    sid = s.get('candidate_id','?')
    sc = s.get('final_score','?')
    rs = s.get('raw_score','?')
    el = s.get('eligible_for_final','?')
    rej = s.get('rejection_reasons',[])
    pt = [p.get('code','') for p in s.get('penalty_trace',[])]
    print(f"  {sid}: raw={rs} final={sc} eligible={el} reject={rej} penalties={pt}")

print()
print("=== FINAL SELECTED ===")
for c in data['final_ranking']['final_selected_clips']:
    print(f"  start={c['start']} score={c['score']} clip_point={c.get('clip_point','')[:80]}")

print()
print("=== QWEN's MODEL PICKS ===")
for c in data['final_ranking'].get('model_final_selected_clips', []):
    print(f"  start={c.get('start')} score={c.get('score')} clip_point={c.get('clip_point','')[:80]}")
