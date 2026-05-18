"""Check what Qwen's model final_selected_clips looks like for comparison."""
import pexpect, json

out = pexpect.run(
    'ssh -o StrictHostKeyChecking=no john@100.97.240.34 cat /home/john/twitch-vod-analyzer/vods/phase4_2770929139/qwen_vision_progressive.json',
    events={'password:': 'Sparky1234\n'},
    timeout=20, encoding='utf-8'
)
idx = out.find('{')
data = json.loads(out[idx:])
model = data['final_ranking'].get('model_final_selected_clips', [])
print("=== QWEN's model_final_selected_clips ===")
for c in model:
    print(json.dumps(c, indent=2))
    print()
print("=== Deterministic final_selected_clips ===")
for c in data['final_ranking']['final_selected_clips']:
    print(f"  start={c['start']} score={c['score']} clip_point={c.get('clip_point','')}")
