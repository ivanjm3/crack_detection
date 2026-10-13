# Stream a Jetson camera to any device over its own Wi-Fi

A step-by-step guide for a team with **one Jetson and one USB camera** that
wants to watch the camera in a browser — laptop or phone — **with no cables,
no router and no campus-network permission**. It is the same method CrackNet
uses to be driven wirelessly.

You will end up with this:

```
   USB camera ──► Jetson ─────────────  Wi-Fi network "TeamCam"  ─────────────►  laptop / phone
                  │  hosts the hotspot (10.42.0.1)                               browser:
                  │  runs stream_server.py (port 8080)                           http://10.42.0.1:8080/
                  └─ comes back by itself after a reboot
```

Time: about 20 minutes. Everything is typed **on the Jetson** (once), then you
never need a cable again.

> The files here: `stream_server.py` (the whole server, standard library +
> OpenCV) and this guide. The server was checked with its `--test-pattern`
> mode (page, snapshot, status and MJPEG stream all answer). The hotspot steps
> are the ones CrackNet runs on a Jetson Orin Nano (JetPack 6, Ubuntu 22.04,
> NetworkManager).

---

## 0. Why a hotspot, and not "everyone on the same Wi-Fi"?

Joining the Jetson and a laptop to the same campus / lab Wi-Fi often **does
not work**, even though both get internet: many networks use *client
isolation*, which blocks every packet between two stations (the router
answers, the other device never does). Nothing on either device can turn that
off.

If the **Jetson itself is the access point**, it owns the network, so nothing
can isolate anything and no one has to approve its MAC address. That is the
whole trick.

> **Quick test whether you even need this.** With both devices on the same
> Wi-Fi, run `ping <jetson-ip>` from the laptop. If it replies, you can skip
> to [§9](#9-alternative-both-devices-already-share-a-network) and just run
> the server. If it times out while the router answers, use the hotspot.

**The cost:** most Jetson Wi-Fi cards are *either* an access point *or* a
client, not both. While the hotspot is up, the **Jetson has no internet**
(see [§8](#8-getting-internet-back-on-the-jetson-apt-pip)). Your laptop is
unaffected only if it has another connection (Ethernet, phone tethering);
otherwise it loses internet while it is on the hotspot.

---

## 1. What you need

| Item | Notes |
|---|---|
| Jetson (Orin Nano / NX / AGX …) with Wi-Fi | The Orin Nano dev kit has an M.2 Wi-Fi card. No Wi-Fi? Use a USB Wi-Fi adapter, or a direct Ethernet cable ([§10](#10-no-wi-fi-card-or-it-cannot-be-an-access-point)). |
| One USB camera (UVC) | Any webcam. Plug it directly into the Jetson. |
| A laptop or phone with Wi-Fi | Any browser. |
| A way to type on the Jetson **once** | Monitor + keyboard, or SSH from a PC ([§2](#2-first-access-to-the-jetson)). |

Check the Wi-Fi card can be an access point (on the Jetson):

```bash
iw list | grep -A8 "Supported interface modes"
```

You need to see `* AP` in the list. If not, see
[§10](#10-no-wi-fi-card-or-it-cannot-be-an-access-point).

---

## 2. First access to the Jetson

Pick whichever you have. You only need it for setup.

| Method | How |
|---|---|
| **Monitor + keyboard** | Plug in, log in, open a terminal. Simplest. |
| **USB cable to a PC** (Jetson dev kit's USB-C/micro-USB device port) | It appears as a network adapter; the Jetson is at **`192.168.55.1`**. `ssh <user>@192.168.55.1` |
| **Ethernet to your router** | `ip -4 addr` on the Jetson shows its address; `ssh <user>@<that address>` |

All commands below are typed in a terminal on the Jetson (or an SSH session
into it).

---

## 3. Check the camera works

```bash
sudo apt-get install -y v4l-utils     # needs internet once; skip if already installed
ls /dev/video*                        # a webcam shows up as /dev/video0 (+ video1 metadata)
v4l2-ctl --list-devices               # confirm it is your camera
v4l2-ctl -d /dev/video0 --list-formats-ext | head -30
```

- Camera number = the `N` in `/dev/videoN` that supports `MJPG` frames; usually
  `0`. If you have several, the server's `--cam N` picks one.
- Python + OpenCV are included in JetPack. Check:
  `python3 -c "import cv2, numpy; print(cv2.__version__)"`. If it fails:
  `sudo apt-get install -y python3-opencv python3-numpy`.

---

## 4. Test the server **before** touching the network

This proves the camera and the code, so that later any problem is a *network*
problem. Copy `stream_server.py` to the Jetson (USB stick, `scp`, or paste it
into `nano`), then:

```bash
mkdir -p ~/cam && cd ~/cam          # put stream_server.py here
python3 stream_server.py --test-pattern
```

No camera needed: it streams a moving test card. In a browser **on the Jetson**
open `http://localhost:8080/` — you should see the card and a line like
`29 fps - 412 frames - up 14 s`. Stop with `Ctrl-C`, then the real camera:

```bash
python3 stream_server.py          # camera 0, 1280x720
# python3 stream_server.py --cam 1 --width 640 --height 480    # other camera / smaller
```

If you see `ERROR: cannot open camera 0`, fix that first (wrong `--cam`,
another program holding it, bad cable). Stop it again with `Ctrl-C`.

---

## 5. Make the Jetson a Wi-Fi hotspot

Choose a name and a password (**8+ characters**). Replace `TeamCam` and
`CHANGE-ME-123` and **do not commit the password anywhere**.

```bash
# the Wi-Fi interface name (usually wlP1p1s0 or wlan0)
IFACE=$(nmcli -t -f DEVICE,TYPE dev status | awk -F: '$2=="wifi"{print $1; exit}')
echo "$IFACE"

# 1. create the access-point profile
sudo nmcli connection add type wifi ifname "$IFACE" con-name team-ap \
     autoconnect yes ssid "TeamCam"

# 2. configure it
sudo nmcli connection modify team-ap \
     802-11-wireless.mode ap \
     802-11-wireless.band bg \
     802-11-wireless.powersave 2 \
     wifi-sec.key-mgmt wpa-psk \
     wifi-sec.proto rsn \
     wifi-sec.pairwise ccmp \
     wifi-sec.group ccmp \
     wifi-sec.psk "CHANGE-ME-123" \
     ipv4.method shared \
     ipv4.addresses 10.42.0.1/24 \
     ipv6.method ignore

# 3. switch it on
sudo nmcli connection up team-ap
```

What the important lines do:

| Setting | Why |
|---|---|
| `802-11-wireless.mode ap` | be the access point |
| `802-11-wireless.band bg` | 2.4 GHz: longest range, supported by every phone and card. (`a` = 5 GHz, if the card does it.) |
| `802-11-wireless.powersave 2` | **radio power-saving off.** Left on, it makes the video stall for hundreds of milliseconds at a time — it looks like "the camera is laggy". |
| `ipv4.method shared` | NetworkManager starts a DHCP/DNS server, so phones and laptops get an address **with nothing else to configure** |
| `ipv4.addresses 10.42.0.1/24` | pins the Jetson's address so the URL never changes |
| `autoconnect yes` | the hotspot comes back after every reboot |

If the Jetson was previously joined to another Wi-Fi, stop it from fighting
the hotspot at boot:

```bash
nmcli -t -f NAME,TYPE connection show | awk -F: '$2=="802-11-wireless"{print $1}'   # list profiles
sudo nmcli connection modify "<your-old-wifi-name>" connection.autoconnect no
```

Check it:

```bash
nmcli -t -f NAME,DEVICE connection show --active      # team-ap  on  <iface>
ip -4 addr show "$IFACE"                              # inet 10.42.0.1/24
```

> **If you are connected over SSH through the same Wi-Fi radio**, bringing the
> hotspot up drops your session (the radio switches role). Do this step from a
> monitor, USB, or Ethernet.

---

## 6. Connect a laptop or phone and open the stream

1. On the laptop/phone, open Wi-Fi settings and join **`TeamCam`** with the
   password you chose. (A phone may say "no internet" — stay connected.)
2. On the Jetson, start the server:
   ```bash
   cd ~/cam && python3 stream_server.py
   ```
3. In the laptop/phone browser: **`http://10.42.0.1:8080/`**

You should see the live video. Other URLs:

| URL | What |
|---|---|
| `http://10.42.0.1:8080/` | page with video + fps line |
| `http://10.42.0.1:8080/stream.mjpg` | raw stream (VLC, OpenCV, an `<img>` tag) |
| `http://10.42.0.1:8080/snapshot.jpg` | one picture |
| `http://10.42.0.1:8080/status.json` | machine-readable status |

SSH also works over the hotspot: `ssh <user>@10.42.0.1`.

Use it from OpenCV on the laptop:

```python
import cv2
cap = cv2.VideoCapture("http://10.42.0.1:8080/stream.mjpg")
ok, frame = cap.read()
```

---

## 7. Start the server automatically at boot

So that "power on" is the only step. Create a service (edit `User=` and the
two paths to match yours):

```bash
sudo tee /etc/systemd/system/team-cam.service >/dev/null <<'EOF'
[Unit]
Description=Jetson camera stream
After=network-online.target NetworkManager.service
Wants=network-online.target

[Service]
User=jetson
WorkingDirectory=/home/jetson/cam
ExecStart=/usr/bin/python3 /home/jetson/cam/stream_server.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now team-cam
systemctl status team-cam --no-pager        # "active (running)"
journalctl -u team-cam -n 20 --no-pager      # the server's own output
```

Reboot to prove it (`sudo reboot`): after about a minute `TeamCam` appears and
the URL works with no one touching the Jetson.

Stop / restart / change flags: `sudo systemctl stop team-cam`,
`sudo systemctl restart team-cam`; flags go on the `ExecStart=` line.

---

## 8. Getting internet back on the Jetson (apt, pip)

While the hotspot is up the Jetson is offline. To install something, swap
temporarily:

```bash
sudo nmcli connection down team-ap
sudo nmcli device wifi connect "<home-or-lab-wifi>" password "<its-password>"
sudo apt-get update && sudo apt-get install -y <package>
# ...then back:
sudo nmcli connection up team-ap
```

Easier: do all installs (§3) **before** creating the hotspot, or plug the
Jetson into **Ethernet** — a wired link works at the same time as the hotspot,
and because the hotspot uses `ipv4.method shared`, phones joined to it get
internet through the Jetson's wired connection too. Nothing to configure.

---

## 9. Alternative: both devices already share a network

If `ping` between the laptop and the Jetson **works** on your Wi-Fi/Ethernet,
you don't need a hotspot:

```bash
ip -4 addr            # on the Jetson: note its address, e.g. 192.168.1.42
python3 stream_server.py
# browser: http://192.168.1.42:8080/
```

The address may change after a reboot (DHCP). Reserve it in the router, or
use the hotspot for a fixed `10.42.0.1`.

---

## 10. No Wi-Fi card, or it cannot be an access point

| Situation | Do this |
|---|---|
| No Wi-Fi in the Jetson | Plug in a **USB Wi-Fi adapter** whose driver lists `* AP` in `iw list` (look this up for the exact model *before* buying; many cheap adapters are client-only), then repeat §5 with its interface name. |
| Card has no `* AP` | Same: use a USB adapter. |
| Don't want Wi-Fi at all | **Direct Ethernet cable** Jetson ⇄ laptop. Set the Jetson wired profile to shared: `sudo nmcli connection modify "Wired connection 1" ipv4.method shared ipv4.addresses 10.42.0.1/24 && sudo nmcli connection up "Wired connection 1"`. The laptop gets an address automatically; open `http://10.42.0.1:8080/`. |
| Dev kit USB-C device port | Already a link at `192.168.55.1` (no setup) — good for development, but it is a cable. |

---

## 11. Plug in your own processing

Everything your project does goes in one function in `stream_server.py`:

```python
def process(frame):
    # frame: BGR numpy array straight from the camera
    # ... run detection, draw boxes, measure something ...
    return frame, {"detections": 3}      # (image to stream, extras for /status.json)
```

Whatever dict you return is merged into `/status.json`, so another device (a
phone alarm page, a logger, a dashboard) can poll `http://10.42.0.1:8080/status.json`
without touching the video. Two lessons from CrackNet:

- **Keep only the newest frame** (the server already does). A slow viewer
  must never make the camera fall behind.
- **Show "no link" distinctly from "all clear".** If you build an alert page,
  a stopped server must not look like "nothing detected".

---

## 12. Troubleshooting

| Symptom | Likely cause → fix |
|---|---|
| `ERROR: cannot open camera 0` | wrong index (`ls /dev/video*`), another program holds it (`sudo fuser /dev/video0`), or unplugged. |
| Page loads, no video | camera opened but not delivering: try `--width 640 --height 480`; try another USB port; check `status.json` `frames` is increasing. |
| `TeamCam` not visible | `nmcli connection up team-ap` errors? Card may not support AP (§1). Check `journalctl -u NetworkManager -n 30`. Some countries' regulatory settings block certain channels: try band `bg`. |
| Joins but no IP / "limited connectivity" | `ipv4.method` must be `shared`; `nmcli connection show team-ap | grep ipv4`. |
| Phone drops off the Wi-Fi after a few seconds | Phones leave networks "without internet" on some versions. Turn off *Smart network switch / Auto-switch to mobile data*, or keep mobile data off. |
| Video stalls every few seconds | radio power-save on: `nmcli connection show team-ap | grep powersave` should say `2`. Also reduce load: `--width 640 --height 480 --quality 70`. |
| Fine on the Jetson's browser, not from the laptop | firewall: `sudo ufw status`; if active, `sudo ufw allow 8080/tcp`. Confirm the laptop is really on `TeamCam`: `ping 10.42.0.1`. |
| SSH session died when creating the hotspot | expected if you were on the same Wi-Fi radio (§5). Reconnect to `10.42.0.1`. |
| Two cameras, one fails | USB 2.0 bandwidth is shared across ports: at 720p+ two MJPEG cameras on one hub often cannot stream at once. Lower resolution, or open them one after the other. |
| Works, then not after reboot | `systemctl is-enabled team-cam`; `nmcli connection show team-ap | grep autoconnect` must be `yes`. |

---

## 13. Command cheat-sheet

```bash
# hotspot
sudo nmcli connection up team-ap          # on
sudo nmcli connection down team-ap        # off
nmcli -t -f NAME,DEVICE connection show --active
ip -4 addr                                # the Jetson's addresses

# server
python3 stream_server.py                  # foreground, Ctrl-C stops
python3 stream_server.py --test-pattern   # no camera needed
sudo systemctl restart team-cam           # if installed as a service
journalctl -u team-cam -f                 # live log

# from the laptop
ping 10.42.0.1
curl http://10.42.0.1:8080/status.json
```

---

### Why each choice was made (so you can change it with confidence)

| Choice | Reason |
|---|---|
| MJPEG, not WebRTC/H.264 | works in any browser with one `<img>` tag, no codec, no JavaScript, trivial to read from OpenCV |
| Camera in `MJPG` mode | raw YUYV at 720p+ over USB 2.0 is a few fps; the camera compresses it for you |
| Standard-library HTTP server | nothing to `pip install` on a Jetson that may be offline |
| Fixed `10.42.0.1` | the URL is always the same, no discovery needed |
| Radio power-save off | otherwise periodic stalls that look like camera lag |
| Latest-frame-only | latency stays constant; a slow viewer cannot build a backlog |
