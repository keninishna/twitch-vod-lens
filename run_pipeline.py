"""Start fresh pipeline run with Bee ready."""
import pexpect, time

p = pexpect.spawn('ssh -o StrictHostKeyChecking=no -o LogLevel=ERROR john@100.97.240.34', timeout=600, encoding='utf-8')
p.expect('password:', timeout=10)
p.sendline('Sparky1234')
time.sleep(2)
p.sendline('')
p.expect('[$#>]', timeout=5)

# Kill any running pipeline
p.sendline('pkill -f "qwen_clip_analyzer" 2>/dev/null; pkill -f "qwen_clip" 2>/dev/null')
p.expect('[$#)>

# Run pipeline
p.sendline('cd ~/twitch-vod-analyzer && PYTHONPATH=. python3 -u src/synthesis/qwen_clip_analyzer_progressive.py --vod-id 2770929139 --skip-audio > /tmp/pipeline_merge.log 2>&1 &')
p.expect('[$#>]', timeout=5)
p.sendline('echo "PID=$!"')
p.expect('[$#>]', timeout=5)
print('PID:', (p.before or '')[:200])

p.sendline('exit')
p.expect(pexpect.EOF, timeout=3)