"""Wait and check pipeline status."""
import sys, time
try:
    import pexpect
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "pexpect", "-q"], capture_output=True)
    import pexpect

for wait_secs, label in [(120, "2 min"), (240, "4 min"), (360, "6 min"), (480, "8 min")]:
    time.sleep(120)
    child = pexpect.spawn(
        'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null john@100.97.240.34',
        timeout=15, encoding='utf-8', echo=False
    )
    child.expect(['password:', pexpect.TIMEOUT], timeout=10)
    child.sendline('Sparky1234')
    child.expect(['[$#]', pexpect.TIMEOUT], timeout=5)

    child.sendline('ps aux | grep "python3.*qwen" | grep -v grep | wc -l')
    child.expect(['[$#]', pexpect.TIMEOUT], timeout=5)
    running = (child.before or "").strip()

    child.sendline('tail -5 /tmp/pipeline_run4.log 2>&1')
    child.expect(['[$#]', pexpect.TIMEOUT], timeout=5)
    tail = (child.before or "")[:500]

    child.sendline('exit')
    child.expect(pexpect.EOF, timeout=3)

    print(f"=== {label} ({wait_secs}s) running={running} ===")
    print(tail)

    if "Results saved" in tail or "Final selection" in tail:
        print("=== COMPLETE! ===")
        break