"""Show full clean clip_point for selected clips."""
import pexpect, json

out = pexpect.run('ssh -o StrictHostKeyChecking=no john@100.97.240.34 cat /home/john/twitch-vod-analyzer/vods/phase4_2770929139/qwen_vision_progressive.json', events={'password:':'Sparky1234\n'}, timeout=20, encoding='utf-8')
d = json.loads(out[out.find('{'):])

for c in d['final_ranking']['final_selected_clips']:
    print(f"start={c['start']} score={c['score']}")
    print(f"  clip_point: {c.get('clip_point','')}")
    print()