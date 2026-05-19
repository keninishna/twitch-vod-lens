"""Check pipeline_v4 tail."""
import pexpect

out = pexpect.run(
    'ssh -o StrictHostKeyChecking=no john@100.97.240.34 wc -c /tmp/pipeline_v4.log',
    events={'password:': 'Sparky1234\n'},
    timeout=15, encoding='utf-8'
)
print('SIZE:', out)

out2 = pexpect.run(
    'ssh -o StrictHostKeyChecking=no john@100.97.240.34 tail -5 /tmp/pipeline_v4.log',
    events={'password:': 'Sparky1234\n'},
    timeout=15, encoding='utf-8'
)
idx = out2.find('[')
print('TAIL:', out2[idx:] if idx >= 0 else out2[:500])