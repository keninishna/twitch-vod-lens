"""SSH into WSL2, start Bee server, run pipeline."""
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

# Check if Bee is running
child.sendline('curl -sS --max-time 3 http://localhost:8082/v1/models 2>&1')
child.expect(['[$#]', pexpect.TIMEOUT], timeout=8)
bee_output = child.before or ""
print("=== BEE CHECK ===")
print(bee_output[:500])

if "model" not in bee_output.lower():
    print("Bee not running. Starting via toggle...")
    child.sendline('~/bee-toggle.sh on 2>&1')
    child.expect(['[$#]', pexpect.TIMEOUT], timeout=120)
    print("=== BEE START ===")
    print((child.before or "")[:2000])

# Run the pipeline with skip-audio
child.sendline('cd ~/twitch-vod-analyzer && PYTHONPATH=. python3 src/synthesis/qwen_clip_analyzer_progressive.py --vod-id 2770929139 --skip-audio 2>&1')
child.expect(['[$#]', pexpect.TIMEOUT], timeout=600)
print("=== PIPELINE OUTPUT (last 15000 chars) ===")
out = child.before or ""
print(out[-15000:])
print("=== END ===")

child.sendline('exit')
child.expect(pexpect.EOF, timeout=5)