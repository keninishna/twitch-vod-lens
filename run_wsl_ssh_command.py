"""SSH into WSL2 with password auth and run commands using pexpect."""
import os, sys, time

PASSWORD = "Sparky1234"
HOST = "john@100.97.240.34"

# Install pexpect if needed
import subprocess
subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "pexpect", "-q"], capture_output=True)

import pexpect

child = pexpect.spawn(f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null {HOST}",
                       timeout=30, encoding='utf-8')

# Wait for password prompt
index = child.expect(["password:", "assword", "[Pp]assword", pexpect.TIMEOUT]], timeout=10)
if index != pexpect.TIMEOUT:
    child.sendline(PASSWORD)
    time.sleep(2)
    child.sendline("cd ~/twitch-vod-analyzer && git pull 2>&1")
    time.sleep(8)
    child.sendline("echo '---DONE---'")
    time.sleep(2)
    output = child.read_nonblockintread_nonblocking(size=10000, timeout=5)
    print("OUTPUT:", output)
else:
    print("TIMEOUT waiting for password prompt")
    print("BEFORE:", child.before)
    print("AFTER:", child.after)
