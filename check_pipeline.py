"""Check pipeline progress and fetch final results."""
import sys, time
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

child.sendline('tail -5 /tmp/pipeline_run4.log 2>&1')
child.expect(['[$#]', pexpect.TIMEOUT], timeout=10)
print("=== LOG TAIL ===")
print((child.before or "")[:2000])

# Check if it's done
child.sendline("grep -c 'Results saved' /tmp/pipeline_run4.log 2>&1")
child.expect(['[$#]', pexpect.TIMEOUT], timeout=5)
res = (child.before or "")
if '1' in res:
    print("\n=== PIPELINE COMPLETE! Fetching results... ===")
    child.sendline(r"""python3 << 'EOF'
import json
with open("/home/john/twitch-vod-analyzer/vods/phase4_2770929139/qwen_vision_progressive.json") as f:
    data = json.load(f)
print("=== ALL CLIP ANALYSES ===")
for c in data.get("clip_details", []):
    a = c.get("analysis", {})
    s = c.get("start", "?")
    cw = a.get("clip_worthiness", "?")
    nt = a.get("narrative_type", "?")
    cp = str(a.get("clip_point", ""))[:120]
    err = a.get("error", "")
    print(f"  {s}s: worth={cw} type={nt} err={err[:40]}")
    print(f"    {cp}")
print()
print("=== SCORED CANDIDATES ===")
for s in data.get("stage2_scored", []):
    sid = s.get("candidate_id", "?")
    sc = s.get("patsfinal_score", "?")
    el = s.get("eligible_for_final", "?")
    rej = s.get("rejection_reasons", [])
    print(f"  {sid}: score={sc} eligible={el} rej={rej}")
print()
fr = data.get("final_ranking", {})
for c in fr.get("final_selected_clips", []):
    s = c.get("start", "?")
    sc = c.get("score", "?")
    cp = str(c.get("clip_point", ""))[:150]
    print(f"SELECTED: start={s} score={sc} title={cp}")
EOF""")
    child.expect(['[$#]', pexpect.TIMEOUT], timeout=15)
    print((child.before or "")[:5000])
else:
    print("\nStill running...")

child.sendline('exit')
child.expect(pexpect.EOF, timeout=5)