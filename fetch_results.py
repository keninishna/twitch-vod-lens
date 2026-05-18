"""Fetch pipeline results from WSL2."""
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

# Read and pretty-print the JSON
SCRIPT = r"""python3 << 'INNEREOF'
import json
with open("/home/john/twitch-vod-analyzer/vods/phase4_2770929139/qwen_vision_progressive.json") as f:
    data = json.load(f)

for c in data.get("clip_details", []):
    a = c.get("analysis", {})
    s = c.get("start", "?")
    cw = a.get("clip_worthiness", "?")
    nt = a.get("narrative_type", "?")
    tr = str(a.get("trigger", ""))[:120]
    pa = str(a.get("payoff", ""))[:120]
    cp = str(a.get("clip_point", ""))[:120]
    print(f"--- Clip at {s}s ---")
    print(f"  clip_worthiness: {cw}/10")
    print(f"  narrative_type: {nt}")
    print(f"  trigger: {tr}")
    print(f"  payoff: {pa}")
    print(f"  clip_point: {cp}")
    print()

fr = data.get("final_ranking", {})
for c in fr.get("final_selected_clips", []):
    s = c.get("start", "?")
    sc = c.get("score", "?")
    cp = str(c.get("clip_point", ""))[:120]
    print(f"SELECTED: start={s}s  score={sc}  title={cp}")

print()
print("GATING:", json.dumps(fr.get("gating_summary", {}), indent=2))
print("FALLBACK:", fr.get("stage3_fallback_activated", "?"))
INNEREOF"""

child.sendline(SCRIPT)
child.expect(['[$#]', pexpect.TIMEOUT], timeout=15)
print("=== RESULTS ===")
print(child.before[:10000] if child.before else "(empty)")

child.sendline('exit')
child.expect(pexpect.EOF, timeout=5)
