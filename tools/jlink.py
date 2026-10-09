#!/usr/bin/env python3
"""Join this Windows machine to the Jetson's Wi-Fi AP, and report the link.

Runs on the HOST (Windows), not on the Jetson.

    python tools/jlink.py join      install the WLAN profile and connect
    python tools/jlink.py status    which route is live, and is SSH up
    python tools/jlink.py leave     disconnect and go back to the other Wi-Fi

WHY THIS EXISTS

The AP password is generated on the Jetson and never typed by anyone. `join`
fetches it over whichever route is currently working (USB, usually), writes a
WLAN profile XML, hands it to `netsh`, and connects. After that Windows
reconnects on its own at every boot, so this is a one-time step per machine.

Installing a profile with `netsh wlan add profile user=current` needs no
administrator rights, which matters because this account does not have them.

Joining the AP takes the Wi-Fi radio away from the infrastructure network.
This machine keeps its internet over Ethernet, so that costs nothing here -
but it is the reason `leave` exists.
"""
import os
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import jssh                                       # noqa: E402  credentials + run

SSID = os.environ.get("CRACKNET_AP_SSID", "CrackNet")
AP_IP = "10.42.0.1"
REMOTE = "/home/sarah/cracknet"

PROFILE = """<?xml version="1.0"?>
<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">
  <name>{ssid}</name>
  <SSIDConfig><SSID><name>{ssid}</name></SSID></SSIDConfig>
  <connectionType>ESS</connectionType>
  <connectionMode>auto</connectionMode>
  <MSM><security>
    <authEncryption>
      <authentication>WPA2PSK</authentication>
      <encryption>AES</encryption>
      <useOneX>false</useOneX>
    </authEncryption>
    <sharedKey>
      <keyType>passPhrase</keyType>
      <protected>false</protected>
      <keyMaterial>{psk}</keyMaterial>
    </sharedKey>
  </security></MSM>
</WLANProfile>
"""


def netsh(*args):
    r = subprocess.run(("netsh",) + args, capture_output=True, text=True)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def fetch_psk():
    """Read the AP password off the Jetson over the live route."""
    c = jssh.connect()
    try:
        chan = c.get_transport().open_session()
        chan.get_pty()
        # Reading the secret needs root; jssh feeds the password to sudo -S.
        # Wrapped in a marker rather than parsed out of the raw reply: the
        # channel has a PTY, so the output also carries command echo and sudo
        # noise, and guessing which line is the key is how you end up
        # installing the word "hidden" as a password.
        chan.exec_command("sudo -S -p '' bash -lc "
                          "'cd %s && NO_COLOR=1 ./netlink.sh psk "
                          "| sed \"s/^/PSKLINE:/\"'" % REMOTE)
        chan.sendall(jssh.PW + "\n")
        out = b""
        while True:
            if chan.recv_ready():
                out += chan.recv(65536)
            elif chan.exit_status_ready():
                while chan.recv_ready():
                    out += chan.recv(65536)
                break
        text = out.decode("utf-8", "replace")
    finally:
        c.close()

    for m in re.finditer(r"PSKLINE:(\S{8,63})\s*$", text, re.M):
        return m.group(1)
    sys.exit("Could not read the AP password. Got:\n" +
             text.replace(jssh.PW, "***"))


def current_ssid():
    _, out = netsh("wlan", "show", "interfaces")
    m = re.search(r"^\s*SSID\s*:\s*(.+)$", out, re.M)
    return m.group(1).strip() if m else None


def cmd_join():
    if current_ssid() == SSID:
        print("already on %s" % SSID)
    else:
        print("reading the AP password off the Jetson ...")
        psk = fetch_psk()

        # Written to a temp file because netsh takes a filename, not a string.
        # Deleted straight after: it holds the key in cleartext.
        path = os.path.join(tempfile.gettempdir(), "cracknet-wlan.xml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(PROFILE.format(ssid=SSID, psk=psk))
        try:
            rc, out = netsh("wlan", "add", "profile",
                            "filename=" + path, "user=current")
        finally:
            os.remove(path)
        if rc != 0:
            sys.exit("netsh could not add the profile:\n" + out)
        print("  profile installed")

        rc, out = netsh("wlan", "connect", "name=" + SSID, "ssid=" + SSID)
        if rc != 0:
            sys.exit("netsh could not connect:\n" + out)
        print("  connecting to %s ..." % SSID)

    # Associating and getting a DHCP lease are not instant, and a probe that
    # fires too early reports a working link as broken.
    for _ in range(30):
        if jssh.reachable(AP_IP, timeout=1.0):
            print("  SSH reachable at %s" % AP_IP)
            return cmd_status()
        time.sleep(1)
    print("  joined, but %s is not answering on port 22 yet." % AP_IP)
    print("  check on the Jetson: sudo ./netlink.sh status")
    return 1


def cmd_leave():
    netsh("wlan", "disconnect")
    print("disconnected from %s" % SSID)
    print("Windows will rejoin its usual network; the Jetson is then only")
    print("reachable over the USB link.")
    return 0


def cmd_status():
    print()
    print("wi-fi        %s" % (current_ssid() or "not connected"))
    for host in jssh.HOSTS:
        state = "reachable" if jssh.reachable(host, timeout=1.5) else "-"
        label = {AP_IP: "jetson AP", "192.168.55.1": "usb link"}.get(host, "route")
        print("%-12s %-15s %s" % (label, host, state))
    print()
    print("dashboard    http://%s:8081/" % (
        AP_IP if jssh.reachable(AP_IP, timeout=1.0) else jssh.HOSTS[-1]))
    return 0


def have_profile():
    rc, out = netsh("wlan", "show", "profiles")
    return rc == 0 and SSID in out


def reconnect(wait=25):
    """Nudge Windows back onto the AP, and wait for SSH to answer.

    Windows marks the profile "connect automatically", and still will not
    reliably rejoin an SSID that vanished while it was connected to it - after
    a Jetson reboot it can sit disconnected for minutes, although an explicit
    connect succeeds at once. Since the whole point of the AP is to work with
    no USB cable attached, the launcher has to do this nudge itself rather
    than leave someone staring at a dead link.
    """
    if not have_profile():
        return False
    sys.stderr.write("link down; reconnecting to %s ...\n" % SSID)
    netsh("wlan", "connect", "name=" + SSID, "ssid=" + SSID)
    for _ in range(wait):
        if jssh.reachable(AP_IP, timeout=1.0):
            sys.stderr.write("  back on %s\n" % AP_IP)
            return True
        time.sleep(1)
    sys.stderr.write("  still no answer from %s\n" % AP_IP)
    return False


def live_host(heal=True):
    for host in jssh.HOSTS:
        if jssh.reachable(host, timeout=1.5):
            return host
    if heal and reconnect():
        return AP_IP
    return None


def cmd_host():
    """Print the live address and nothing else, for the batch launcher.

    Progress goes to stderr so the batch file can capture stdout verbatim.
    """
    host = live_host()
    print(host or jssh.HOSTS[-1])      # a guess beats an empty URL
    return 0 if host else 1


def main():
    cmd = (sys.argv[1] if len(sys.argv) > 1 else "status").lower()
    return {"join": cmd_join, "leave": cmd_leave,
            "status": cmd_status, "host": cmd_host}.get(
        cmd, lambda: sys.exit(__doc__))()


if __name__ == "__main__":
    sys.exit(main() or 0)
