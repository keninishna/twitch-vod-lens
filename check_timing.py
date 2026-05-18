"""Check response times and errors."""
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

child.sendline("grep 'Response in' /tmp/pipeline_run3.log")
child.expect(['[$#]', pexpect.TIMEOUT], timeout=10)
print("=== RESPONSE TIMES ===")
print((child.before or "")[:2000])

child.sendline("grep -i 'timeout\\|error\\|fail\\|WARN' /tmp/pipeline_run3.log")
child.expect(['[$#]', pexpect.TIMEOUT], timeout=10)
print("=== ERRORS ===")
print((child.before or "")[:2000])

child.sendline('exit')
child.expect(pexpect.EOF, timeout=5)