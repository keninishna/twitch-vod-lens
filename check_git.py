"""Verify fix on WSL2."""
import pexpect
out = pexpect.run('ssh -o StrictHostKeyChecking=no john@100.97.240.34 grep startswith /home/john/twitch-vod-analyzer/src/synthesis/qwen_clip_analyzer_progressive.py', events={'password:': 'Sparky1234\n'}, timeout=15, encoding='utf-8')
print(out[:500])
