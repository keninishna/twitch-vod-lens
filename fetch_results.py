"""Wait for pipeline, then fetch detailed results including response times."""
import sys, time
try:
    import pexpect
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "pexpect", "-q"], capture_output=True)
    import pexpect

child = pexpect.spawn(
    'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null john@100.97.240.34',
    timeout=900, encoding='utf-8', echo=False
)
child.expect(['password:', pexpect.TIMEOUT], timeout=10)
child.sendline('Sparky1234')
time.sleep(2)
child.sendline('')
child.expect('[$#>]', timeout=5)

# Wait for completion by polling the log
for attempt in range(10):
    time.sleep(60)
    child.sendline("grep -c 'Results saved' /tmp/pipeline_run5.log")
    child.expect('[$#>]', timeout=5)
    out = child.before or ""
    if '1' in out.split('\n')[-2:]:
        print(f"=== COMPLETE after {attempt+1} min ===")
        break
    else:
        print(f"  Waiting... ({attempt+1} min)")

# Get response times
child.sendline("grep 'Response in' /tmp/pipeline_run5.log")
child.expect('[$#>]', timeout=5)
print("=== RESPONSE TIMES ===")
print(child.before[:1000] if child.before else "")

# Get scored candidates
child.sendline(r"""python3 << 'EOF'
import json
with open("/home/john/twitch-vod-analyzer/vods/phase4_2770929139/qwen_vision_progressive.json") as f:
    data = json.load(f)
print("=== CLIP ANALYSES ===")
for c in data.get("clip_details", []):
    a = c.get("analysis", {})
    s = c.get("start", "?")
    cw = a.get("clip_worthiness", "?")
    nt = a.get("narrative_type", "?")
    cp = str(a.get("clip_point", ""))[:120]
    print(f"  {s}s: worth={cw} type={nt}")
    print(f"    {cp}")
print()
print("=== SCORED CANDIDATES ===")
for s in data.get("stage2_scored", []):
    sid = s.get("candidate_id", "?")
    sc = s.get("final_score", "?")
    el = s.get("eligible_for_final", "?")
    rej = s.get("rejection_reasons", [])
    pt = s.get("penalty_trace", [])
    codes = [p["code"] for p in pt]
    print(f"  {sid}: score={sc} eligible={el}")
    print(f"    penalties={codes} rejects={rej}")
print()
fr = data.get("final_ranking", {})
for c in fr.get("final_selected_clips", []):
    s = c.get("start", "?")
    sc = c.get("score", "?")
    cp = str(c.get("clip_point", ""))[:150]
    print(f"SELECTED: start={s} score={sc} title={cp}")
EOF""")
child.expect('[$#>]', timeout=15)
print(child.before[:6000] if child.before else "")

child.sendline('exit')
child.expect(pexpect.EOF, timeout=3)