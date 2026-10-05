#!/usr/bin/env python3
"""Run commands on the Jetson over SSH, from the host machine.

Lives here rather than in a scratch directory because it gets rewritten from
scratch every time a session's temp folder is cleared. It runs on the HOST
(Windows/Linux laptop), not on the Jetson.

Credentials come from the environment so they stay out of git:

    # PowerShell
    $env:JETSON_HOST = "192.168.55.1"; $env:JETSON_USER = "sarah"; $env:JETSON_PASS = "..."
    # bash
    export JETSON_HOST=192.168.55.1 JETSON_USER=sarah JETSON_PASS=...

or put them in a .jetson.env file next to this script (gitignored):

    JETSON_HOST=192.168.55.1
    JETSON_USER=sarah
    JETSON_PASS=...

Usage
    python jssh.py "nvpmodel -q"                    run a command
    python jssh.py --sudo "nvpmodel -m 2"           run it as root
    python jssh.py --file local.sh                  upload a script and run it
    python jssh.py --put local.py /remote/path.py   copy a file up
    python jssh.py --get /remote/f.jpg local.jpg    copy a file down
    python jssh.py --timeout 1800 "bash build.sh"   longer timeout (default 600 s)

Requires paramiko:  pip install paramiko

Note for Git Bash on Windows: POSIX paths in arguments get mangled into Windows
paths before Python sees them, which makes a remote path like /home/sarah/x.py
fail with "No such file". Prefix the command with MSYS_NO_PATHCONV=1.
"""
import os
import sys

try:
    import paramiko
except ImportError:
    sys.exit("paramiko is not installed.  pip install paramiko")

REMOTE_TMP = "/tmp/.jssh_script.sh"


def credentials():
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".jetson.env")
    values = {}
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    values[k.strip()] = v.strip()

    host = os.environ.get("JETSON_HOST", values.get("JETSON_HOST", "192.168.55.1"))
    user = os.environ.get("JETSON_USER", values.get("JETSON_USER", "sarah"))
    pw = os.environ.get("JETSON_PASS", values.get("JETSON_PASS"))
    if not pw:
        sys.exit("No password. Set JETSON_PASS, or create tools/.jetson.env "
                 "(see the docstring).")
    return host, user, pw


HOST, USER, PW = credentials()


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PW, timeout=20,
              look_for_keys=False, allow_agent=False)
    return c


def run(c, cmd, use_sudo, timeout):
    chan = c.get_transport().open_session()
    chan.settimeout(timeout)
    chan.get_pty()
    chan.exec_command("sudo -S -p '' " + cmd if use_sudo else cmd)
    if use_sudo:
        chan.sendall(PW + "\n")          # sudo reads the password from stdin
    out = b""
    while True:
        if chan.recv_ready():
            out += chan.recv(65536)
        elif chan.exit_status_ready():
            while chan.recv_ready():
                out += chan.recv(65536)
            break
    rc = chan.recv_exit_status()
    sys.stdout.write(out.decode("utf-8", "replace").replace(PW, "***"))
    print(f"\n[exit {rc}]")
    return rc


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)

    use_sudo, timeout = False, 600
    while args and args[0] in ("--sudo", "--timeout"):
        if args[0] == "--sudo":
            use_sudo, args = True, args[1:]
        else:
            timeout, args = int(args[1]), args[2:]

    c = connect()
    try:
        if args[0] == "--put":
            s = c.open_sftp(); s.put(args[1], args[2]); s.close()
            print(f"put {args[1]} -> {args[2]}")
        elif args[0] == "--get":
            s = c.open_sftp(); s.get(args[1], args[2]); s.close()
            print(f"get {args[1]} -> {args[2]}")
        elif args[0] == "--file":
            with open(args[1], "r", encoding="utf-8", newline="") as f:
                body = f.read().replace("\r\n", "\n")   # CRLF would break bash
            s = c.open_sftp()
            with s.open(REMOTE_TMP, "w") as rf:
                rf.write(body)
            s.close()
            sys.exit(run(c, f"bash {REMOTE_TMP}", use_sudo, timeout))
        else:
            sys.exit(run(c, "bash -lc " + repr(args[0]), use_sudo, timeout))
    finally:
        c.close()


if __name__ == "__main__":
    main()
