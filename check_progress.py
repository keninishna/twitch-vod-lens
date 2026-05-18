"""Check pipeline merge.log progress."""
import pexpect, time

out = pexpect.run(
    'ssh -o StrictHostKeyChecking=no john@100.97.240.34 tail -20 /tmp/pipeline_final.log',
    events={'password:': 'Sparky1234\n'},
    timeout=20, encoding='utf-8'
)
print(out[:4000] if out else '')