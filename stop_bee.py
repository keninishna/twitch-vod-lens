"""Verify Bee is stopped."""
import pexpect, time
p = pexpect.spawn('ssh -o StrictHostKeyChecking=no -o LogLevel=ERROR john@100.97.240.34', timeout=15, encoding='utf-8')
p.expect('password:', timeout=10)
p.sendline('Sparky1234')
time.sleep(1)
p.sendline('')
p.expect('[$#>]', timeout=5)

# Try nvidia-smi
p.sendline('nvidia-smi --query-gpu=memory.used,memory.total,processes.used --format=csv,noheader 2>&1 || echo "no nvidia-smi"')
p.expect('[$#>]', timeout=10)
out = p.before or ""
print("GPU:", out[-500:] if len(out) > 500 else out)

p.sendline('ps aux | grep -E "(llama|bee)" | grep -v grep | head -3')
p.expect('[$#>]', timeout=5)
out = p.before or ""
print("PROCS:", out[-500:] if len(out) > 500 else out)

p.sendline('exit')
p.expect(pexpect.EOF, timeout=3)
