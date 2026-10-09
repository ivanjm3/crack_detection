# Reaching the Jetson without a USB cable

The goal: start something on the Jetson once, then unplug USB and keep driving
it from the Windows PC over the network.

The short version: **the Jetson hosts its own Wi-Fi access point**, the Windows
PC joins it, and `cracknet.bat` works unchanged. One command on the Jetson, one
command on the PC, both one-time — the link then comes back by itself on every
reboot.

```
cracknet netmode ap     # on the Jetson: become an access point   (one time)
cracknet join           # on this PC: join it                     (one time)
cracknet live           # unplug USB; everything still works
```

---

## 1. Why the obvious answer does not work

The obvious answer is "put both machines on the campus Wi-Fi and SSH across
it". Both machines *are* on it, both reach the internet, and it still cannot
work. Two independent walls, each measured rather than assumed:

### IOT-PROJ has client isolation

| From | To | Result |
|---|---|---|
| Windows `172.16.61.172` | Jetson `172.16.61.136` | ping 100% loss, TCP 22 refused |
| Windows | gateway `172.16.61.1` | **0% loss, 1 ms** |
| Jetson | internet (`connectivity-check.ubuntu.com`) | **HTTP 204 in 0.55 s** |
| Jetson | Windows, the gateway, `8.8.8.8` by ping | all fail (ICMP blocked too) |

Same `/24`, both healthy, both online — and zero packets between them. The
gateway answers instantly while the other station never does, which is the
signature of client isolation. It is an access-point setting; nothing on either
machine can turn it off.

### The wired jack will not issue a lease

`enP8p1s0` shows `carrier=1` — the cable is in and the link is up. DHCP then
times out twice over 45 s each. The Windows PC gets `172.16.79.74` from the
same wired network, so the network is fine; the port only serves **registered
MAC addresses**, and the Jetson's is not one of them.

A static address on that subnet might have slipped through, and is not worth
doing: a /23 belonging to someone else, with unknown allocations, is not a
place to invent an IP.

### So the Jetson stops being a client

As an AP, the Jetson **owns its subnet**: there is no other access point to
isolate anything, and no DHCP server that has to approve us. The Windows PC
joins and keeps its own internet over Ethernet.

---

## 2. What it costs

The radio is a Realtek `rtl88x2ce`. It reports `AP` among its supported modes
but advertises **no valid interface combinations**, so it does AP *or* client,
never both.

**In `ap` mode the Jetson has no internet.** That is fine for running and
viewing CrackNet and wrong for `apt-get`, so the mode is switchable:

```bash
sudo ./netlink.sh ap      # access point, reachable from the PC, no internet
sudo ./netlink.sh wifi    # rejoin IOT-PROJ, internet, unreachable from the PC
sudo ./netlink.sh status  # which one is live, and every usable address
sudo ./netlink.sh boot ap # which one comes back after a reboot
```

Switching is also available as `cracknet netmode ap|wifi` from Windows.

> `netmode` is the one command that cannot safely travel over the link it
> changes — tearing the AP down also destroys the path its reply would take.
> The launcher checks for the USB link first and, if it is absent, warns and
> asks before continuing.

---

## 3. The pieces

| Where | File | Does |
|---|---|---|
| Jetson | `jetson/netlink.sh` | Creates and switches the AP; `boot` sets what survives a reboot |
| Windows | `tools/jlink.py` | `join`, `leave`, `status`, `host` |
| Windows | `tools/jssh.py` | Tries each route in `JETSON_HOSTS` and uses the first that answers |
| Windows | `cracknet.bat` | `join`, `leave`, `link`, `netmode` |

### The AP

| | |
|---|---|
| SSID | `CrackNet` |
| Address | **`10.42.0.1`** — pinned, so the launcher can rely on it |
| Security | WPA2-PSK / CCMP |
| Band | 2.4 GHz by default (`CRACKNET_AP_BAND=a` for 5 GHz) |
| DHCP | NetworkManager `ipv4.method shared`, i.e. its own dnsmasq |
| Password | **generated at random on first use**, never stored in this repo |
| Powersave | Disabled — on the radio it shows up as multi-hundred-ms stalls in the MJPEG stream, which reads as "the camera is laggy" rather than as a network setting |

Nobody ever types the password. `cracknet join` reads it off the Jetson over
whatever route currently works, writes a WLAN profile, and hands it to `netsh`
— which needs no administrator rights, and this account does not have them.

### Routes are probed, not configured

`tools/.jetson.env` lists them in order:

```
JETSON_HOSTS=10.42.0.1,192.168.55.1
```

Each is tried with a 1.5 s TCP probe and the first to answer is used, so
plugging or unplugging USB needs no edit anywhere. A dead candidate costs
1.5 s instead of SSH's 20 s timeout, which is what makes probing affordable on
every command.

---

## 4. The reboot test, and what it caught

Claiming "it comes back by itself" is only worth anything if the board has been
rebooted, so it was.

**The Jetson side passed.** `cracknet-ap` has `autoconnect yes` with priority
20 and `IOT-PROJ` has `autoconnect no`, and after a reboot the AP was up at
`10.42.0.1` with no intervention. Nothing needs starting.

**The Windows side failed.** Windows sat `disconnected` for over two minutes,
even though the profile reads `Connection mode: Connect automatically`. An
explicit `netsh wlan connect` then succeeded instantly. Windows does not retry
an SSID promptly after it vanishes mid-connection.

That is exactly the situation in which USB is unplugged, so the launcher heals
it: when no route answers, `jlink.py` reconnects to `CrackNet` itself and waits
for SSH, printing `link down; reconnecting to CrackNet ...` so the pause is
explained rather than mysterious.

Verified from a deliberately disconnected adapter: `cracknet check` reconnected
and ran, with the Jetson reporting the session arriving from `10.42.0.163` over
`wlP1p1s0`.

---

## 5. If you are locked out

The failure that matters is the AP not coming up with no cable attached. In
order:

1. **USB cable.** `192.168.55.1` is still in `JETSON_HOSTS`, so plugging USB in
   makes everything work again with no changes.
2. `netlink.sh ap` **restores the client Wi-Fi's autoconnect if the AP fails to
   start**, so a board that cannot host an AP rejoins IOT-PROJ rather than
   coming back with no network at all.
3. `cracknet link` says which routes are live before anything else is tried.

> The AP is WPA2 with a random password, but SSH there is still password
> authentication, and CrackNet's own server has no authentication and binds
> `0.0.0.0`. Anyone on the `CrackNet` SSID can watch the dashboard.
