"""Check pipeline done.log results."""
import pexpect

out = pexpect.run(
    'ssh -o StrictHostKeyChecking=no john@100.97.240.34 tail -30 /tmp/pipeline_done.log',
    events={'password:': 'Sparky1234\n'}, timeout=20, encoding='utf-8')
print(out[-2500:])
