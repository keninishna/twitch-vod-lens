"""SSH into WSL2 with password auth and run git pull."""
import os, sys, time, subprocess
from subprocess import Popen, PIPE, STDOUT

HOST = "john@100.97.240.34"
PASSWORD = "Sparky1234"

# Build the ssh command
cmd = ["ssh", "-o", "StrictHostKeyChecking=no",
       "-o", "UserKnownHostsFile=/dev/null",
       "-o", "PreferredAuthentications=password",
       HOST]

p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=STDOUT, text=True, bufsize=0)

# Wait a moment then send password
time.sleep(1.5)

# Send password
p.stdin.write(PASSWORD + "\n")
p.stdin.flush()
time.sleep(1.5)

# Run commands
p.stdin.write("cd ~/twitch-vod-analyzer && git pull 2>&1\n")
p.stdin.flush()
time.sleep(8)

p.stdin.write("echo '====SSH_DONE===='\n")
p.stdin.flush()
time.sleep(2)

# Read all output
try:
    out, _ = p.communicate(timeout=5)
except:
    out = ""
    # Try reading what we can
    import select
    while True:
        r, _, _ = select.select([p.stdout], [], [], 2)
        if r:
            chunk = os.read(p.stdout.fileno(), 4096)
            if not chunk:
                break
            out += chunk.decode(errors='replace')
        else:
            break

print(out)
print("EXIT:", p.returncode if p.poll() is not None else "still running")