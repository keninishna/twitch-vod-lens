"""Verify fix on WSL2 and test locally."""
import pexpect, json

# Check commit on WSL2
out = pexpect.run(
    'ssh -o StrictHostKeyChecking=no john@100.97.240.34 cd /home/john/twitch-vod-analyzer && git log --oneline -1',
    events={'password:': 'Sparky1234\n'},
    timeout=15, encoding='utf-8'
)
print('Commit:', out.split()[-2] if len(out.split()) > 2 else out[:100])

# Test the fix logic locally
from src.synthesis.qwen_clip_analyzer_progressive import safe_json_parse

# Exact case: doubled braces as seen in 758s response
test = '{{\n  "clip_start": 758,\n  "clip_end": 878,\n  "clip_worthiness": 8,\n  "narrative_type": "storytelling"\n}}'
r = safe_json_parse(test)
print(f'Test doubled braces: {r}')
print(f'  clip_worthiness={r["clip_worthiness"] if r else "FAILED"}')
