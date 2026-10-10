#!/usr/bin/env python3
"""Give the Jetson this PC's internet, with no admin rights and no ICS.

Runs on the HOST (Windows), not on the Jetson.

    python tools/jproxy.py "sudo apt-get update"
    python tools/jproxy.py "pip3 install --user gz-something"
    python tools/jproxy.py --selftest

HOW

Windows only forwards packets between adapters when Internet Connection
Sharing is on, and turning that on needs administrator rights this account
does not have. This does not forward packets at all.

It opens the usual SSH session to the Jetson and asks sshd for a *remote*
port forward: the Jetson listens on 127.0.0.1:3128, and every connection made
to that port is carried back down the SSH session to THIS process, which acts
as a small HTTP / HTTPS (CONNECT) proxy and opens the real connection from
here, using this PC's own internet.

    Jetson apt/pip/curl -> 127.0.0.1:3128 -> [ssh session] -> this PC -> internet

Because the SSH session is opened outbound from Windows, no firewall rule is
needed either - the Windows firewall never sees an inbound connection. It works
over whichever route jssh finds first (the CrackNet AP or USB).

The tunnel exists only while the command runs. Nothing is written into the
Jetson's apt or pip configuration, so a board that is later used without this
tool does not have a dead proxy configured.

LIMITS

HTTP and HTTPS only. Tools that ignore proxy settings (raw git over ssh://,
some installers that open their own sockets) will not go through it.
"""
import os
import select
import socket
import sys
import threading
import time
from urllib.parse import urlsplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PORT = 3128
MAX_HEADER = 65536
IDLE = 120          # seconds a tunnelled connection may sit silent


# ------------------------------------------------------------ the proxy core
class _Endpoint:
    """Uniform recv/send over a paramiko Channel or a plain socket."""

    def __init__(self, sock):
        self.s = sock

    def recv(self, n=65536):
        return self.s.recv(n)

    def send(self, b):
        self.s.sendall(b)

    def close(self):
        try:
            self.s.close()
        except Exception:
            pass


def _read_head(ep):
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = ep.recv()
        if not chunk:
            return None, b""
        buf += chunk
        if len(buf) > MAX_HEADER:
            return None, b""
    head, _, rest = buf.partition(b"\r\n\r\n")
    return head, rest


def _pipe(a, b):
    """Copy both directions until either side closes or goes idle."""
    done = threading.Event()

    def pump(src, dst):
        try:
            src.s.settimeout(IDLE)
            while not done.is_set():
                data = src.recv()
                if not data:
                    break
                dst.send(data)
        except Exception:
            pass
        finally:
            done.set()

    t = threading.Thread(target=pump, args=(b, a), daemon=True)
    t.start()
    pump(a, b)
    done.set()
    t.join(timeout=2)


def handle(client_sock):
    """Serve one proxied connection arriving from the Jetson."""
    client = _Endpoint(client_sock)
    upstream = None
    try:
        head, rest = _read_head(client)
        if head is None:
            return
        lines = head.split(b"\r\n")
        try:
            method, target, version = lines[0].decode("latin-1").split(" ", 2)
        except ValueError:
            client.send(b"HTTP/1.1 400 Bad Request\r\n\r\n")
            return

        if method.upper() == "CONNECT":
            host, _, port = target.rpartition(":")
            upstream = socket.create_connection((host, int(port or 443)), 15)
            client.send(b"HTTP/1.1 200 Connection established\r\n\r\n")
            if rest:
                upstream.sendall(rest)
        else:
            # Plain HTTP arrives as an absolute-URI request line. Re-issue it
            # as an origin-form request and ask the server to close, so one
            # proxied connection is exactly one exchange.
            u = urlsplit(target)
            host, port = u.hostname, u.port or 80
            if not host:
                client.send(b"HTTP/1.1 400 Bad Request\r\n\r\n")
                return
            path = (u.path or "/") + ("?" + u.query if u.query else "")
            kept = [h for h in lines[1:]
                    if not h.lower().startswith((b"proxy-", b"connection:"))]
            req = ("%s %s %s\r\n" % (method, path, version)).encode("latin-1")
            req += b"\r\n".join(kept) + b"\r\nConnection: close\r\n\r\n"
            upstream = socket.create_connection((host, port), 15)
            upstream.sendall(req + rest)

        _pipe(client, _Endpoint(upstream))
    except Exception as exc:
        try:
            client.send(("HTTP/1.1 502 Bad Gateway\r\n\r\n%s" % exc).encode())
        except Exception:
            pass
    finally:
        client.close()
        if upstream is not None:
            try:
                upstream.close()
            except Exception:
                pass


class _ChanSock:
    """Give a paramiko Channel the sendall/settimeout surface handle() uses."""

    def __init__(self, chan):
        self.c = chan

    def recv(self, n):
        return self.c.recv(n)

    def sendall(self, b):
        self.c.sendall(b)

    def settimeout(self, t):
        self.c.settimeout(t)

    def close(self):
        self.c.close()


# ------------------------------------------------------------------ self-test
def selftest():
    """Exercise the proxy core over loopback, with no Jetson involved."""
    import urllib.request

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    port = srv.getsockname()[1]

    def accept_loop():
        while True:
            try:
                c, _ = srv.accept()
            except OSError:
                return
            threading.Thread(target=handle, args=(c,), daemon=True).start()

    threading.Thread(target=accept_loop, daemon=True).start()

    proxy = "http://127.0.0.1:%d" % port
    opener = urllib.request.build_opener(urllib.request.ProxyHandler(
        {"http": proxy, "https": proxy}))
    failed = 0
    for label, url in (
            ("http  (plain)", "http://connectivity-check.ubuntu.com/"),
            ("https (CONNECT)", "https://connectivity-check.ubuntu.com/")):
        try:
            r = opener.open(url, timeout=20)
            ok = r.status in (200, 204)
            print("  %-16s %s %s" % (label, r.status, "ok" if ok else "UNEXPECTED"))
            failed += 0 if ok else 1
        except Exception as exc:
            print("  %-16s FAIL %s" % (label, exc))
            failed += 1
    srv.close()
    return failed


# --------------------------------------------------------------- the tunnel
def run(command, use_sudo=False, timeout=1800):
    import jssh

    c = jssh.connect()
    t = c.get_transport()
    t.set_keepalive(20)

    def on_connection(chan, origin, server):
        threading.Thread(target=handle, args=(_ChanSock(chan),),
                         daemon=True).start()

    try:
        t.request_port_forward("127.0.0.1", PORT, handler=on_connection)
    except Exception as exc:
        sys.exit("The Jetson refused the port forward on %d: %s\n"
                 "  (is another tunnel already running, or is "
                 "AllowTcpForwarding off in sshd?)" % (PORT, exc))

    proxy = "http://127.0.0.1:%d" % PORT
    env = ("http_proxy={p} https_proxy={p} HTTP_PROXY={p} HTTPS_PROXY={p} "
           "no_proxy=localhost,127.0.0.1").format(p=proxy)
    # apt reads Acquire::*::Proxy before the environment, and ignores the
    # environment entirely under sudo's env_reset, so pass it explicitly.
    apt = ("-o Acquire::http::Proxy={p} -o Acquire::https::Proxy={p}"
           .format(p=proxy))
    # A leading "sudo" is accepted and handled here, because a PTY command that
    # prompts for a password would just hang with nobody to answer it.
    if command.startswith("sudo "):
        use_sudo, command = True, command[5:]
    cmd = command.replace("apt-get ", "apt-get %s " % apt, 1) \
        if "apt-get " in command else command
    if use_sudo:
        full = "sudo -S -p '' env %s bash -lc %r" % (env, cmd)
    else:
        full = "bash -lc %r" % ("export " + env + "; " + cmd)

    print("tunnel up via %s  (jetson 127.0.0.1:%d -> this PC)" % (jssh.HOST, PORT))
    chan = t.open_session()
    chan.settimeout(timeout)
    chan.get_pty()
    chan.exec_command(full)
    if use_sudo:
        chan.sendall(jssh.PW + "\n")

    out = b""
    started = time.time()
    while True:
        if chan.recv_ready():
            data = chan.recv(65536)
            sys.stdout.write(data.decode("utf-8", "replace").replace(jssh.PW, "***"))
            sys.stdout.flush()
        elif chan.exit_status_ready():
            while chan.recv_ready():
                sys.stdout.write(chan.recv(65536).decode("utf-8", "replace")
                                 .replace(jssh.PW, "***"))
            break
        else:
            time.sleep(0.05)
        if time.time() - started > timeout:
            print("\n[timed out after %d s]" % timeout)
            break
    rc = chan.recv_exit_status()
    try:
        t.cancel_port_forward("127.0.0.1", PORT)
    except Exception:
        pass
    c.close()
    print("\n[exit %d - tunnel closed]" % rc)
    return rc


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    if args[0] == "--selftest":
        print("proxy core, loopback only:")
        n = selftest()
        print("  %s" % ("all passed" if n == 0 else "%d FAILED" % n))
        return 1 if n else 0
    use_sudo = False
    if args[0] == "--sudo":
        use_sudo, args = True, args[1:]
    return run(args[0], use_sudo)


if __name__ == "__main__":
    sys.exit(main())
