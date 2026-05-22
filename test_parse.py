"""Write test script to WSL2 and run it."""
import pexpect, time

p = pexpect.spawn('ssh -o StrictHostKeyChecking=no -o LogLevel=ERROR john@100.97.240.34', timeout=20, encoding='utf-8')
p.expect('password:', timeout=10)
p.sendline('Sparky1234')
time.sleep(2)
p.sendline('')
p.expect('[$#>]', timeout=5)

# Write test script using heredoc
p.sendline("cat > /tmp/test_parse.py << 'ENDPARSE'")
p.sendline('import sys, json')
p.sendline("sys.path.insert(0, '/home/john/twitch-vod-analyzer')")
p.sendline('from src.synthesis.qwen_clip_analyzer_progressive import safe_json_parse')
p.sendline('')

p.sendline('# Test doubled braces')
p.sendline("test = '{{\\n  \\\"clip_start\\\": 758,\\n  \\\"test\\\": true\\n}}'")
p.sendline("r = safe_json_parse(test)")
p.sendline("print('Result:', repr(r))")
p.sendline('')

p.sendline('# Test normal JSON')
p.sendline("r.sendline("r = safe_json_parse('{\"a\":: 1}')")
p.sendline("print('Normal:', repr(r))")
p.sendline('ENDPARSE')
p.expect('Time:     ', timeout=5)

p.sendline('python3 /tmp/test_parse.py')
time.sleep(2)
try:
    output = p.read_nonblocking(size=5000, timeout=5)
    print('OUTPUT:', output[-2000:])
except:
    pass

p.sendline('exit')
p.expect(pexpect.EOF, timeout=3)