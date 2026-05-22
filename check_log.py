"""Check pipeline log directly."""
import pexpect
out = pexpect.run(
    'ssh -o StrictHostKeyChecking=no john@100.97.240.34 tail -5 /tmp/pipeline_live.log',
    events={'password:': 'Sparky1234\n'}, timeout=15, encoding='utf-8')
print(out[:2000])
