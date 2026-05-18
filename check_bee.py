"""Check Bee status on WSL2."""
import pexpect, time

out = pexpect.run(
    'ssh -o StrictHostKeyChecking=no -o LogLevel=ERROR john@100.97.240.34 curl -s --max-time 5 http://localhost:8082/v1/models 2>&1',
    events={'password:': 'Sparky1234\n'},
    timeout=20, encoding='utf-8'
)
# Strip password prompt noise
idx = out.find('{') if out else -1
if idx >= 0:
    print('Bee is UP:', out[idx:idx+200])
else:
    print('Bee DOWN or not ready')
    print('Got:', out[:300] if out else 'empty')