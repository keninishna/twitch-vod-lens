"""Quick status check on WSL2 pipeline."""
import sys
try:
    import pexpect
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "pexpect", "-q"], capture_output=True)
    import pexpect

child = pexpect.spawn(
    'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null john@100.97.240.34',
    timeout=30, encoding='utf-8', echo=False
)
child.expect(['password:', pexpect.TIMEOUT], timeout=10)
child.sendline('Sparky1234')
child.expect(['[$#]', pexpect.TIMEOUT], timeout=10)

child.sendline('ps aux | grep "python3.*qwen" | grep -v grep 2>&1')
child.expect(['[$#]', pexpect.TIMEOUT], timeout=5)
print("=== PROCESS ===")
print((child.before or "")[:500])

child.sendline('tail -20 /tmp/pipeline_run3.log 2>&1')
child.expect(['[$#]', pexpect.TIMEOUT], timeout=10)
print("=== LOG ===")
print((child.before or "")[:3000])

child.sendline('exit')
child.expect(pexpect.EOF, timeout=5)