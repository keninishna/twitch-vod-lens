"""Debug merge matching - check model_final_selected_clips starts."""
import pexpect, json

out = pexpect.run(
    'ssh -o StrictHostKeyChecking=no john@100.97.240.34 cat /home/john/twitch-vod-analyzer/vods/phase4_2770929139/qwen_vision_progressive.json',
    events={'password:': 'Sparky1234\n'},
    timeout=20, encoding='utf-8'
)
idx = out.find('{')
data = json.loads(out[idx:])

print("=== model_final_selected_clips starts ===")
model = data['final_ranking'].get('model_final_selected_clips', [])
for c in model:
    print(f"  Qwen start={c.get('start')} trim_start={c.get('suggested_trim_start')} end={c.get('end')}")
print()
print("=== deterministic final_selected_clips starts ===")
for c in data['final_ranking']['final_selected_clips']:
    print(f"  Det start={c.get('start')} trim_start={c.get('suggested_trim_start')} end={c.get('end')} score={c.get('score')}")