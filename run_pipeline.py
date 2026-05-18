"""Kill old pipeline, restart with unbuffered Python, monitor."""
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

# Kill old process
child.sendline('kill 24541 2>&1; sleep 1; ps aux | grep qwen_clip | grep -v grep')
child.expect(['[$#]', pexpect.TIMEOUT], timeout=10)
print("=== KILL ===")
print((child.before or "")[:500])

# Run with unbuffered Python, save to log
child.sendline('cd ~/twitch-vod-analyzer && PYTHONPATH=. python3 -u src/synthesis/qwen_clip_analyzer_progressive.py --vod-id 2770929139 --skip-audio > /tmp/pipeline_run3.log 2>&1 &')
child.expect(['[$#]', pexpect.TIMEOUT], timeout=5)
child.sendline('echo "STARTED: PID=$!"')
child.expect(['[$#]', pexpect.TIMEOUT], timeout=5)
print((child.before or "")[:500])

# Wait and check progress periodically
for check in range(6):
    time.sleep(120)  # 2 min intervals
    child.sendline('tail -5 /tmp/pipeline_run3.log 2>&1')
    child.expect(['[$#]', pexpect.TIMEOUT], timeout=10)
    output = (child.before or "")[:1000]
    # Check if pipeline completed
    if "Results saved" in output or "Final selection" in output:
        print(f"=== CHECK {check+1} (COMPLETE!) ===")
        print(output)
        break
    else:
        print(f"=== CHECK {check+1} (12 min total) ===")
        print(output)

# Final tail
child.sendline('echo "=== LOG END ===" && wc -l /tmp/pipeline_run3.log')
child.expect(['[$#]', pexpect.TIMEOUT], timeout=10)
print("=== FINAL ===")
print((child.before or "")[:500])

child.sendline('exit')
child.expect(pexpect.EOF, timeout=5)