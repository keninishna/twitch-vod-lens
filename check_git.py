"""Simple git status - send return key to clear warning."""
import sys, time
try:
    import pexpect
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "pexpect", "-q"], capture_output=True)
    import pexpect

child = pexpect.spawn(
    'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null john@100.97.240.34',
    timeout=15, encoding='utf-8', echo=False
)
child.expect(['password:', pexpect.TIMEOUT], timeout=10)
child.sendline('Sparky1234')
time.sleep(2)
# Send return to clear any warning prompt
child.sendline('')
child.expect('[$#>]', timeout=5)
child.sendline('cd ~/twitch-vod-analyzer && git log --oneline -1')
child.expect('[$#>]', timeout=5)
print("=== GIT ===")
print(child.before[:800] if child.before else "")

child.sendline('grep "clamp" src/synthesis/scoring.py | head -3')
child.expect('[$#>]', timeout=5)
print("=== CLAMP CHECK ===")
print(child.before[:800] if child.before else "")

child.sendline('exit')
child.expect(pexpect.EOF, timeout=3)