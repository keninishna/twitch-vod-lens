"""Check 758s raw response."""
import pexpect, json

out = pexpect.run(
    'ssh -o StrictHostKeyChecking=no john@100.97.240.34 cat /home/john/twitch-vod-analyzer/vods/phase4_2770929139/qwen_vision_progressive.json',
    events={'password:': 'Sparky1234\n'},
    timeout=20, encoding='utf-8'
)
idx = out.find('{')
d = json.loads(out[idx:])

for c in d['clip_details']:
    if c.get('start') == 758:
        a = c['analysis']
        print('keys:', list(a.keys()))
        raw = a.get('raw', '')
        print('raw length:', len(raw))
        print('raw first 50:', repr(raw[:50]))
        print('raw last 50:', repr(raw[-50:]))
        break