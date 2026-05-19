"""Check pipeline_v3 progress."""
import pexpect

out = pexpect.run(
    'ssh -o StrictHostKeyChecking=no john@100.97.240.34 tail -5 /tmp/pipeline_v3.log',
    events={'password:': 'Sparky1234\n'},
    timeout=20, encoding='utf-8'
)
idx = out.find('[')
print(out[idx:] if idx >= 0 else out[:500])