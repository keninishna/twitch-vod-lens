"""Check pipeline_v2.log status."""
import pexpect

out = pexpect.run(
    'ssh -o StrictHostKeyChecking=no john@100.97.240.34 wc -c /tmp/pipeline_v2.log',
    events={'password:': 'Sparky1234\n'},
    timeout=20, encoding='utf-8'
)
print('SIZE:', out[:100])

out2 = pexpect.run(
    'ssh -o StrictHostKeyChecking=no john@100.97.240.34 tail -5 /tmp/pipeline_v2.log',
    events={'password:': 'Sparky1234\n'},
    timeout=20, encoding='utf-8'
)
idx = out2.find('[')
print('TAIL:', out2[idx:] if idx >= 0 else out2[:500])