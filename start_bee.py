"""Start Bee server on WSL2."""
import pexpect, time

p = pexpect.spawn('ssh -o StrictHostKeyChecking=no -o LogLevel=ERROR john@100.97.240.34', timeout=120, encoding='utf-8')
p.expect('password:', timeout=10)
p.sendline('Sparky1234')
time.sleep(2)
p.sendline('')
p.expect('[$#>]', timeout=5)

p.sendline('~/bee-toggle.sh on 2>&1')
p.expect('[$#>]', timeout=120)
print((p.before or '')[:2000])

p.sendline('exit')
p.expect(pexpect.EOF, timeout=3)
