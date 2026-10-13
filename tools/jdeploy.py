"""Copy the project's Jetson-side files to the board, in one SSH session.

    python tools/jdeploy.py            upload what the server and launcher need
    python tools/jdeploy.py --all      every jetson/*.py, *.sh plus both pages

Text files are sent with LF endings: a CRLF in a shell script is
`bash: \\r: command not found` on the Jetson, and Windows checkouts can have
them whatever the repo says. Shell scripts are made executable.

It only copies. It does not restart anything - run `cracknet live` (or
`scan`) afterwards, because a running Python server keeps the old code.
"""
import posixpath
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jssh                                                  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/home/sarah/cracknet"

# What the console actually runs, plus the two pages it serves.
CORE = [
    "jetson/serve_scan.py", "jetson/serve.py", "jetson/scanner.py",
    "jetson/cracknet.sh", "jetson/netlink.sh",
    "viewer/dashboard.html", "viewer/alarm.html",
]


def files(everything):
    if not everything:
        return [ROOT / p for p in CORE]
    out = sorted((ROOT / "jetson").glob("*.py")) + \
          sorted((ROOT / "jetson").glob("*.sh"))
    return out + [ROOT / "viewer" / "dashboard.html",
                  ROOT / "viewer" / "alarm.html"]


def main():
    todo = files("--all" in sys.argv[1:])
    missing = [p for p in todo if not p.exists()]
    if missing:
        sys.exit("missing: " + ", ".join(str(m.relative_to(ROOT)) for m in missing))

    c = jssh.connect()
    try:
        sftp = c.open_sftp()
        for p in todo:
            data = p.read_bytes().replace(b"\r\n", b"\n")
            dest = posixpath.join(REMOTE, p.name)
            with sftp.open(dest, "wb") as f:
                f.write(data)
            if p.suffix == ".sh":
                sftp.chmod(dest, 0o755)
            print("  %-22s -> %s  (%d bytes)" % (p.name, dest, len(data)))
        sftp.close()
    finally:
        c.close()
    print("pushed %d file(s). Restart to use them: cracknet live | scan" % len(todo))


if __name__ == "__main__":
    main()
