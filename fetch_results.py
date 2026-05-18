"""Fetch results of pipeline run with 180s timeout."""
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

print("=== ALL CLIP ANALYSES ===")
for c in data.get("clip_details", []):
    a = c.get("analysis", {})
    s = c.get("start", "?")
    cw = a.get("clip_worthiness", "?")
    nt = a.get("narrative_type", "?")
    err = a.get("error", "")
    cp = str(a.get("clip_point", ""))[:120]
    print(f"Clip {s}s: worth={cw} type={nt} err={err[:50]}")
    print(f"  title: {cp}")
    print()

print("=== SCORED CANDIDATES ===")
for s in data.get("stage2_scored", []):
    sid = s.get("candidate_id", "?")
    sc = s.get("final_score", "?")
    el = s.get("eligible_for_final", "?")
    rej = s.get("rejection_reasons", [])
    print(f"{sid}: score={sc} eligible={el} rej={rej}")

print()
fr = data.get("final_ranking", {})
for c in fr.get("final_selected_clips", []):
    s = c.get("start", "?")
    sc = c.get("score", "?")
    cp = str(c.get("clip_point", ""))[:150]
    print(f"SELECTED: start={s} score={sc} title={cp}")
print("GATING:", json.dumps(fr.get("gating_summary", {}), indent=2))
EOF"""

child.sendline(SCRIPT)
child.expect(['[$#]', pexpect.TIMEOUT], timeout=15)
print((child.before or "")[:8000])

child.sendline('exit')
child.expect(pexpect.EOF, timeout=5)