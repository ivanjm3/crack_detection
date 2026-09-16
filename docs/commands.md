# Operating the Jetson from this Windows machine

Everything below is run in **PowerShell on the Windows PC**. The Jetson is
reached over SSH; nothing here needs a monitor or keyboard attached to the board.

---

## 0. Connection facts

| | |
| --- | --- |
| User | `sarah` |
| Password | (ask — not stored in this repo) |
| **USB link address** | **`192.168.55.1`** — stable, always works when the USB cable is connected |
| Wi-Fi address | `172.16.61.175` — DHCP lease on `IOT-PROJ`, **changes**; confirm before relying on it |
| Project directory | `/home/sarah/cracknet` |
| Viewer port | `8080` |

Prefer `192.168.55.1`. It is a direct link between the two machines and does not
depend on the campus network. The Wi-Fi address is only needed when viewing the
stream from a *different* device (a phone, another laptop).

> The Jetson's **ethernet port does not work** — the link comes up at 1000 Mbps
> but DHCP never answers, because the MAC (`3c:6d:66:bf:3a:b2`) is not registered
> on the campus network. Wi-Fi is the working path. See `WORKLOG-2026-09-16.md` §1.

---

## 1. Connect

```powershell
ssh sarah@192.168.55.1
```

Type the password when prompted. You land in `/home/sarah`; the project is in
`cracknet/`.

```bash
cd ~/cracknet
```

To run one command without staying logged in:

```powershell
ssh sarah@192.168.55.1 "cd ~/cracknet && ./cracknet.sh status"
```

Note the quoting: the whole remote command is one double-quoted PowerShell
string. Use single quotes *inside* it if the command itself needs quotes.

---

## 2. Start and stop the system

`cracknet.sh` is the single entry point. It runs preflight checks, sets the
performance mode, locks the camera controls and starts the viewer.

```powershell
ssh sarah@192.168.55.1 "cd ~/cracknet && ./cracknet.sh start"
ssh sarah@192.168.55.1 "cd ~/cracknet && ./cracknet.sh stop"
ssh sarah@192.168.55.1 "cd ~/cracknet && ./cracknet.sh restart"
ssh sarah@192.168.55.1 "cd ~/cracknet && ./cracknet.sh status"
ssh sarah@192.168.55.1 "cd ~/cracknet && ./cracknet.sh check"
```

| Command | Does |
| --- | --- |
| `start [flags]` | preflight → MAXN_SUPER → camera lock → server. Waits for the first real frame before reporting success, so "up" means up. |
| `stop` | SIGTERM, then SIGKILL if needed, then confirms `/dev/video0` was released |
| `restart [flags]` | `stop` then `start` |
| `status` | pid, uptime, port, current detection, fps, exposure state, URLs |
| `check` | preflight only — changes nothing |
| `logs` | follows `serve.log` (Ctrl-C to leave) |

Any `serve.py` flag passes straight through:

```powershell
ssh sarah@192.168.55.1 "cd ~/cracknet && ./cracknet.sh start --rotate 180"
ssh sarah@192.168.55.1 "cd ~/cracknet && ./cracknet.sh start --thresh 0.65 --min-area 400"
ssh sarah@192.168.55.1 "cd ~/cracknet && ./cracknet.sh restart --no-auto-exposure"
```

**`start` warns about sudo and carries on.** Setting the power mode needs root,
and it will not hang waiting for a password. If you see
`no passwordless sudo - skipping nvpmodel/jetson_clocks`, set it once per boot:

```powershell
ssh -t sarah@192.168.55.1 "sudo nvpmodel -m 2 && sudo jetson_clocks"
```

`-t` allocates a terminal so the sudo prompt is visible. Mode **2** is
`MAXN_SUPER`; mode 0 is 15 W on this board and would *reduce* performance.

---

## 3. View the stream

Once `start` reports success, open in a browser **on this PC**:

```
http://192.168.55.1:8080/
```

From a phone or another laptop on the `IOT-PROJ` network, use the Wi-Fi address
that `status` prints.

Fetch a single frame or the machine-readable state without a browser:

```powershell
Invoke-WebRequest http://192.168.55.1:8080/snapshot.jpg -OutFile frame.jpg
Invoke-WebRequest http://192.168.55.1:8080/status.json -UseBasicParsing |
    Select-Object -ExpandProperty Content
```

Watch the detection state live from PowerShell:

```powershell
while ($true) {
    (Invoke-WebRequest http://192.168.55.1:8080/status.json -UseBasicParsing).Content
    Start-Sleep -Seconds 2
}
```

| Endpoint | Returns |
| --- | --- |
| `/` | viewer page with live status |
| `/stream.mjpg` | raw MJPEG stream (also plays in VLC) |
| `/snapshot.jpg` | most recent frame |
| `/status.json` | coverage, fps, inference time, exposure, gain, clipping |

---

## 4. Deploy code changes

Edit files in `jetson\` on this PC, then copy them over. **Always edit here, not
on the Jetson** — the repo is the source of truth and the two copies are kept
identical.

```powershell
cd C:\Users\student\Desktop\rp_proj
scp jetson\live.py sarah@192.168.55.1:~/cracknet/
scp jetson\*.py jetson\*.sh sarah@192.168.55.1:~/cracknet/
```

Restart afterwards so the new code is actually loaded:

```powershell
ssh sarah@192.168.55.1 "cd ~/cracknet && ./cracknet.sh restart"
```

Verify the two copies match:

```powershell
$local = Get-ChildItem jetson -File | ForEach-Object {
    "{0}  {1}" -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower(), $_.Name
} | Sort-Object
$remote = (ssh sarah@192.168.55.1 "cd ~/cracknet && sha256sum *.py *.sh *.md | sort -k2") -split "`n" |
    Where-Object { $_ -match '^[0-9a-f]{64}' } | ForEach-Object { $_.Trim() } | Sort-Object
if (Compare-Object $local $remote) { "MISMATCH" } else { "in sync" }
```

Copy results back:

```powershell
scp sarah@192.168.55.1:~/cracknet/run1.mp4 .
scp -r sarah@192.168.55.1:~/cracknet/val_samples .
```

---

## 5. Diagnostics

Run these when something looks wrong. All are read-only.

```powershell
# is the camera there, and on the right format?
ssh sarah@192.168.55.1 "v4l2-ctl --list-devices; v4l2-ctl -d /dev/video0 --get-fmt-video"

# current camera controls - exposure should be servoing, gain should be low
ssh sarah@192.168.55.1 "v4l2-ctl -d /dev/video0 --list-ctrls | grep -E 'exposure|gain|focus'"

# power mode and clocks
ssh sarah@192.168.55.1 "nvpmodel -q; sudo -n jetson_clocks --show 2>/dev/null | head -3"

# network, and whether the Wi-Fi address has changed
ssh sarah@192.168.55.1 "ip -4 -brief addr; nmcli -t -f DEVICE,STATE,CONNECTION device status"

# temperature and load
ssh sarah@192.168.55.1 "cat /sys/devices/virtual/thermal/thermal_zone*/temp; uptime; free -h"

# server log
ssh sarah@192.168.55.1 "tail -30 ~/cracknet/serve.log"
```

Live system monitor (interactive, needs `-t`):

```powershell
ssh -t sarah@192.168.55.1 "sudo jtop"
```

---

## 6. Rebuild and re-validate

Only needed after a JetPack upgrade, a new ONNX export, or if an engine is
missing or refuses to load.

```powershell
# rebuild both engines - takes ~6 minutes, stop the server first
ssh sarah@192.168.55.1 "cd ~/cracknet && ./cracknet.sh stop && bash build_engines.sh"

# TensorRT vs ONNX parity - expect under 0.1% mask disagreement
ssh sarah@192.168.55.1 "cd ~/cracknet && python3 check_parity.py"

# exposure control law - expect 7/7
ssh sarah@192.168.55.1 "cd ~/cracknet && python3 test_ae_sim.py"

# blob geometry behind the shape-filter thresholds
ssh sarah@192.168.55.1 "cd ~/cracknet && python3 analyze_blobs.py"

# capture fresh frames for tuning
ssh sarah@192.168.55.1 "cd ~/cracknet && ./cracknet.sh stop && python3 capture_samples.py --n 20"
```

> Engines are tied to **this board and this TensorRT version**. Never copy an
> engine between machines — rebuild it.

---

## 7. Running the detector without the server

`cracknet.sh` starts the network viewer. To watch the console output instead —
useful when tuning, because it prints per-frame:

```powershell
ssh -t sarah@192.168.55.1 "cd ~/cracknet && ./cracknet.sh stop && python3 live.py --headless"
```

Ctrl-C to stop. Use `-t` so Ctrl-C reaches the remote process. Add `--record
run.mp4` to save video.

The camera can only be opened by one process at a time, hence the `stop` first.

---

## 8. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `ssh: connect to host ... Connection refused` | USB cable unplugged, or the Jetson is off. Check `ping 192.168.55.1`. |
| `cannot open /dev/video0` | Another process holds the camera. `./cracknet.sh stop`, or `ssh sarah@192.168.55.1 "fuser -k /dev/video0"`. |
| `camera NOT PRESENT` in preflight | C920 unplugged, or plugged into a hub. Use a USB-A port directly on the board. |
| Server starts then dies | `./cracknet.sh logs`. Usually a missing engine or the camera being held. |
| Stream loads but the image is sideways | `./cracknet.sh restart --rotate 90` (or 180 / 270). |
| Everything reads `CRACK` on an ordinary scene | Expected off-pavement — the model cannot say "not pavement". Raise `--thresh` / `--min-area`, or see `perception.html` §5. |
| Image too dark or too bright | Check `exposure`/`gain` in `status`. If gain is at 160 the scene is genuinely dark. Adjust with `--ae-target`. |
| Wi-Fi URL unreachable from a phone | Lease changed. Re-read it from `./cracknet.sh status`. |
| `sudo` prompt hangs a command | Add `-t` to the ssh invocation. |
| Low fps | Capture tops out around 22 fps (CPU MJPEG decode). Long exposure also caps it: 500 = 50 ms ⇒ 20 fps max. |
| apt fails, "Release file is not valid yet" | Clock drifted; NTP is blocked on this network. See `WORKLOG-2026-09-16.md` §1 for the HTTP-header fix. |

---

## 9. Git, on this PC

Git is installed but **not on PATH**. Either use the full path or add it for the
session:

```powershell
$env:Path += ";C:\Users\student\AppData\Local\Programs\Git\cmd"
cd C:\Users\student\Desktop\rp_proj
git status
git add -A; git commit -m "..."; git push
```

Remote: `https://github.com/ivanjm3/crack_detection`.

Model weights and TensorRT engines are excluded by `.gitignore` — they live in
`crack_outputs\` on disk and on the Jetson, not in the repo.

> PowerShell 5.1 has no `&&` operator. Chain with `;`, or use
> `cmd1; if ($?) { cmd2 }` when the second should only run on success. The `&&`
> in the *remote* commands above is fine — that runs in bash on the Jetson.
