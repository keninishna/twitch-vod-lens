"""Check raw clip_point for 758s to find the title issue."""
import pexpect, json

out = pexpect.run(
    'ssh -o StrictHostKeyChecking=no john@100.97.240.34 cat /home/john/twitch-vod-analyzer/vods/phase4_2770929139/qwen_vision_progressive.json',
    events={'password:': 'Sparky1234\n'}, timeout=20, encoding='utf-8')
idx = out.find('{')
d = json.loads(out[idx:])

for c in d['final_ranking']['final_selected_clips']:
    if c.get('clip_point', ''):
        print(f"start={c['start']}")
        print(f"  clip_point: {repr(c.get('clip_point',''))}")
        print(f"  model clip_point from qwen: ", end="")
        # Check below...")

# Also check Qwen's model_final_selected_clips
print("\n=== Qwen's model clip_points ===")
for mc in d['final_ranking'].get('model_final_selected_clips', []):
    print(f"  start={mc.get('start')}: {repr(mc.get('clip_point',''))}")

# Also check the deterministic title_dedup.py's title
print("\n=== Stage 1 analysis clip_points (for 758s) ===")
for c in d['clip_details']:
    if c.get('start') == 758:
        print(f"  Stage 1 clip_point: {repr(c.get('analysis',{}).get('clip_point',''))}")