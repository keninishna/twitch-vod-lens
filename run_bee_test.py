"""Start Bee, pull latest, run pipeline, check results."""
import pexpect, time

p = pexpect.spawn('ssh -o StrictHostKeyChecking=no -o LogLevel=ERROR john@100.97.240.34', timeout=900, encoding='utf-8')
p.expect('password:', timeout=10)
p.sendline('Sparky1234')
time.sleep(2)
p.sendline('')
p.expect('[$#>]', timeout=5)

# Start Bee
p.sendline('~/bee-toggle.sh on 2>&1')
p.expect('[$#>]', timeout=120)
print('BEE:', (p.before or '')[-300:])

# Pull latest code
p.sendline('cd ~/twitch-vod-analyzer && git fetch origin && git reset --hard origin/main 2>&1')
p.expect('[$#>]', timeout=15)
p.sendline('git log --oneline -1')
p.expect('[$#>]', timeout=5)
print('GIT:', (p.before or '')[-100:])

# Clear cache and run
p.sendline('rm -rf __pycache__ src/__pycache__ src/synthesis/__pycache__ src/synthesis/schemas/__pycache__ 2>/dev/null')
p.expect('[$#>]', timeout=5)
p.sendline('cd ~/twitch-vod-analyzer && PYTHONPATH=. python3 -u src/synthesis/qwen_clip_analyzer_progressive.py --vod-id 2770929139 --skip-audio > /tmp/pipeline_fix.log 2>&1 &')
p.expect('[$#>]', timeout=5)
p.sendline('echo "PID=\\$!"')
p.expect('[$#>]', timeout=5)
print('PID:', (p.before or '')[-100:])

p.sendline('exit')
p.expect(pexpect.EOF, timeout=3)