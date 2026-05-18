"""Kill old, pull new, run pipeline on WSL2."""
import sys, time
try:
    import pexpect
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "pexpect", "-q"], capture_output=True)
    import pexpect

child = pexpect.spawn(
    'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null john@100.97.240.34',
    timeout=20, encoding='utf-8', echo=False
)
child.expect(['password:', pexpect.TIMEOUT], timeout=10)
child.sendline('Sparky1234')
child.expect(['[$#]', pexpect.TIMEOUT], timeout=10)

# Kill old pipeline processes
child.sendline('pkill -f "qwen_clip_analyzer_progressive" 2>&1; sleep 1; echo "KILLED"')
child.expect(['[$#]', pexpect.TIMEOUT], timeout=10)
print(child.before[:500] if child.before else "")

# Pull latest
child.sendline('cd ~/twitch-vod-analyzer && git pull 2>&1')
child.expect(['[$#]', pexpect.TIMEOUT], timeout=15)
print((child.before or "")[:1000])

# Run pipeline with unbuffered output
child.sendline('cd ~/twitch-vod-analyzer && PYTHONPATH=. python3 -u src/synthesis/qwen_clip_analyzer_progressive.py --vod-id 2770929139 --skip-audio > /tmp/pipeline_run4.log 2>&1 &')
child.expect(['[$#]', pexpect.TIMEOUT], timeout=5)
child.sendline('echo "PID=$!"')
child.expect(['[$#]', pexpect.TIMEOUT], timeout=5)
print("STARTED:", child.before[:500] if child.before else "")

child.sendline('exit')
child.expect(pexpect.EOF, timeout=5)