"""Check pipeline result - specifically 758s clip."""
import pexpect, json

out = pexpect.run(
    'ssh -o StrictHostKeyChecking=no john@100.97.240.34 cat /home/john/twitch-vod-analyzer/vods/phase4_2770929139/qwen_vision_progressive.json',
    events={'password:': 'Sparky1234\n'},
    timeout=20, encoding='utf-8'
)
idx = out.find('{')
d = json.loads(out[idx:])

for c in d['clip_details']:
    s = c.get('start')
    a = c['analysis']
    cw = a.get('clip_worthiness', a.get('clip_worthy', '?'))
    nt = a.get('narrative_type', '?')
    err = a.get('error', '')
    cp = str(a.get('clip_point', ''))[:80]
    print(f'Clip {s}s: worth={cw} type={nt} err={err[:30]}')
    if err:
        raw = a.get('raw', '')
        print(f'  raw_start: {raw[:60]}...')
    else:
        print(f'  clip_point: {cp}')