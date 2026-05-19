"""Check git version on WSL2."""
import pexpect

out = pexpect.run(
    'ssh -o StrictHostKeyChecking=no john@100.97.240.34 cd /home/john/twitch-vod-analyzer && git log --oneline -3',
    events={'password:': 'Sparky1234\n'},
    timeout=15, encoding='utf-8'
)
idx = out.find('90d38ed' if '90d38ed' in out else out.find('589dd94'))
print(out[:500] if out else 'empty')

out2 = pexpect.run(
    'ssh -o StrictHostKeyChecking=no john@100.97.240.34 grep -c "clamped_trim_start" /home/john/twitch-vod-analyzer/src/synthesis/schemas/clip_intelligence_stages.py',
    events={'password:': 'Sparky1234\n'},
    timeout=15, encoding='utf-8'
)
print('CLAMPED IN SCHEMA:', out2[:100])