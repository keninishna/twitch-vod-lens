"""Fetch full analysis and criticism fields for all clips."""
import sys
try:
    import pexpect
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "pexpect", "-q"], capture_output=True)
    import pexpect

child = pexpect.spawn(
    'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null john@100.97.240.34',
    timeout=30, encoding='utf-8', echo=False
)
child.expect(['password:', pexpect.TIMEOUT], timeout=10)
child.sendline('Sparky1234')
child.expect(['[$#]', pexpect.TIMEOUT], timeout=10)

SCRIPT = r"""python3 << 'EOF'
import json
with open("/home/john/twitch-vod-analyzer/vods/phase4_2770929139/qwen_vision_progressive.json") as f:
    data = json.load(f)

for c in data.get("clip_details", []):
    a = c.get("analysis", {})
    s = c.get("start", "?")
    print(f"=== Clip {s}s ===")
    print(f"  worthiness: {a.get('clip_worthiness', '?')}")
    print(f"  narrative_type: {a.get('narrative_type', '?')}")
    print(f"  has_narrative_payoff: {a.get('has_narrative_payoff', '?')}")
    print(f"  requires_context: {a.get('requires_context', '?')}")
    print(f"  transactional_reaction: {a.get('transactional_reaction', '?')}")
    print(f"  suggested_trim_start: {a.get('suggested_trim_start', '?')}")
    print(f"  suggested_trim_end: {a.get('suggested_trim_end', '?')}")
    print(f"  trim_start_reason: {str(a.get('trim_start_reason', ''))[:80]}")
    print(f"  trim_end_reason: {str(a.get('trim_end_reason', ''))[:80]}")
    print(f"  clip_point: {str(a.get('clip_point', ''))[:120]}")
    print(f"  cand_start: {c.get('start')} cand_end: {c.get('end')}")
    print()
EOF"""

child.sendline(SCRIPT)
child.expect(['[$#]', pexpect.TIMEOUT], timeout=15)
print((child.before or "")[:8000])

child.sendline('exit')
child.expect(pexpect.EOF, timeout=5)