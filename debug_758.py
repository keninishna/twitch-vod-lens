"""Check raw response and error for 758s clip."""
import pexpect, json

# Get all analyses and find the 758s one with error/raw fields
out = pexpect.run(
    'ssh -o StrictHostKeyChecking=no john@100.97.240.34 cat /home/john/twitch-vod-analyzer/vods/phase4_2770929139/qwen_vision_progressive.json',
    events={'password:': 'Sparky1234\n'},
    timeout=20, encoding='utf-8'
)
idx = out.find('{')
d = json.loads(out[idx:])

for c in d['clip_details']:
    a = c.get('analysis', {})
    s = c.get('start')
    cw = a.get('clip_worthiness', a.get('clip_worthy', '?'))
    err = a.get('error', '')
    raw = a.get('raw', '')
    if s == 758 or err or raw:
        print(f"Clip {s}s:")
        print(f"  clip_worthiness: {cw}")
        print(f"  error: {err[:200]}")
        print(f"  raw: {raw[:300]}")
        print(f"  all keys: {list(a.keys())}")
        print()