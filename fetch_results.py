"""Check raw Qwen responses for empty analysis clips."""
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

SCRIPT = """python3 << 'EOF'
import json
with open("/home/john/twitch-vod-analyzer/vods/phase4_2770929139/qwen_vision_progressive.json") as f:
    data = json.load(f)

for c in data.get("clip_details", []):
    a = c.get("analysis", {})
    s = c.get("start", "?")
    # Check for error message
    err = a.get("error", "")
    clip_w = a.get("clip_worthiness", a.get("clip_worthy", "MISSING"))
    print(f"Clip {s}s: worthiness={clip_w} error={err}")
    # Show all keys in analysis
    keys = list(a.keys())
    print(f"  keys: {keys}")
    # If no clip_worthiness, check for raw field
    raw = a.get("raw", "")[:200]
    if raw:
        print(f"  raw: {raw}")
    print()
EOF"""

child.sendline(SCRIPT)
child.expect(['[$#]', pexpect.TIMEOUT], timeout=15)
print("=== RAW ANALYSIS CHECK ===")
print((child.before or "")[:5000])

child.sendline('exit')
child.expect(pexpect.EOF, timeout=5)