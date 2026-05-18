"""SSH into WSL2, run the pipeline with wait_for_bee_api preflight."""
import sys, time
try:
    import pexpect
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "pexpect", "-q"], capture_output=True)
    import pexpect

child = pexpect.spawn(
    'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null john@100.97.240.34',
    timeout=600, encoding='utf-8', echo=False
)
child.expect(['password:', pexpect.TIMEOUT], timeout=10)
child.sendline('Sparky1234')
child.expect(['[$#]', pexpect.TIMEOUT], timeout=10)

child.sendline('cd ~/twitch-vod-analyzer && git pull 2>&1')
child.expect(['[$#]', pexpect.TIMEOUT], timeout=15)
print("=== GIT PULL ===")
print((child.before or "")[:1000])

# Run pipeline with skip-audio and new preflight
child.sendline('cd ~/twitch-vod-analyzer && PYTHONPATH=. python3 src/synthesis/qwen_clip_analyzer_progressive.py --vod-id 2770929139 --skip-audio 2>&1')
child.expect(['[$#]', pexpect.TIMEOUT], timeout=600)
print("=== PIPELINE OUTPUT (last 10000 chars) ===")
out = child.before or ""
print(out[-10000:])
print("=== END ===")

child.sendline('exit')
child.expect(pexpect.EOF, timeout=5)