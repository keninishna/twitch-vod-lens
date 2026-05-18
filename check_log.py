"""Fetch pipeline results from WSL2 via SSH using pexpect."""
import sys, time, subprocess

# Ensure pexpect is installed
subprocess.run([sys.executable, "-m", "pip", "install", "pexpect", "-q"], capture_output=True)
import pexpect

def ssh_command(cmd, timeout=10):
    """Run a command on WSL2 and return output."""
    child = pexpect.spawn(
        'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null john@100.97.240.34',
        timeout=timeout, encoding='utf-8', echo=False
    )
    child.expect(['password:', pexpect.TIMEOUT], timeout=10)
    child.sendline('Sparky1234')
    time.sleep(2)
    child.sendline('')
    child.expect('[$#>]', timeout=5)
    child.sendline(cmd)
    child.expect('[$#>]', timeout=timeout)
    output = child.before or ""
    child.sendline('exit')
    child.expect(pexpect.EOF, timeout=3)
    return output

# Check log tail
log_tail = ssh_command("tail -30 /tmp/pipeline_run5.log", timeout=10)
print("=== LOG TAIL ===")
print(log_tail[:3000])

# Get response times
resp = ssh_command("grep 'Response in' /tmp/pipeline_run5.log", timeout=10)
print("\n=== RESPONSE TIMES ===")
print(resp[:1000])

# Read the JSON results
import json as j
SCRIPT = r"""python3 << 'EOF'
import json
with open("/home/john/twitch-vod-analyzer/vods/phase4_2770929139/qwen_vision_progressive.json") as f:
    data = json.load(f)
print("=== CLIPS ===")
for c in data.get("clip_details", []):
    a = c.get("analysis", {})
    s = c.get("start", "?")
    cw = a.get("clip_worthiness", "?")
    cp = str(a.get("clip_point", ""))[:100]
    print(f"  {s}s: worth={cw}  {cp}")
print()
print("=== SCORED ===")
for s in data.get("stage2_scored", []):
    sid = s.get("candidate_id", "?")
    sc = s.get("final_score", "?")
    el = s.get("eligible_for_final", "?")
    rej = s.get("rejection_reasons", [])
    pt = [p["code"] for p in s.get("penalty_trace", [])]
    print(f"  {sid}: score={sc} eligible={el}")
    print(f"    penalties={pt} rejects={rej}")
print()
fr = data.get("final_ranking", {})
for c in fr.get("final_selected_clips", []):
    print(f"SELECTED: start={c.get('start')} score={c.get('score')} title={str(c.get('clip_point',''))[:120]}")
EOF"""
results = ssh_command(SCRIPT, timeout=15)
print("\n=== RESULTS ===")
print(results[:5000])
