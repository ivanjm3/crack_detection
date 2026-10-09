# RUN — copy-paste commands

Everything here runs in **PowerShell on the Windows PC**. Nothing is typed on
the Jetson, and no USB cable is needed.

Open PowerShell and go to the project once per session:

```powershell
cd C:\Users\student\Desktop\rp_proj
```

> The `.\` prefix is required in PowerShell. In `cmd.exe` drop it.
> Or just **double-click `cracknet.bat`** for a menu.

---

## Start it

```powershell
.\cracknet.bat link
```

```powershell
.\cracknet.bat live
```

`live` runs preflight, applies `jetson_clocks`, locks exposure on every camera,
starts the server, waits for the port, then asks `Open it now? [Y/n]` — press
Enter.

Dashboard:

```
http://10.42.0.1:8081/
```

**Plug the three cameras in first.** `live` refuses to start after a failed
preflight. If it stops with a list of checks, that list *is* the answer.

---

## Everyday

| Want | Command |
|---|---|
| Live preview, ~10 fps | `.\cracknet.bat live` |
| Survey scan, ~6.5 s/position | `.\cracknet.bat scan` |
| Is it running? | `.\cracknet.bat status` |
| Why won't it start? | `.\cracknet.bat logs` |
| Stop, release the cameras | `.\cracknet.bat stop` |
| Reopen the dashboard | `.\cracknet.bat open` |
| Check hardware only | `.\cracknet.bat check` |

```powershell
.\cracknet.bat status
```

```powershell
.\cracknet.bat stop
```

`live` and `scan` **restart** rather than start, so running one twice is safe.

---

## Network link

```powershell
.\cracknet.bat link
```

Safe to run any time — a read-only probe. Before unplugging USB you want:

```
wi-fi        CrackNet
jetson AP    10.42.0.1       reachable
```

`10.42.0.1` is tried first, so commands already go over Wi-Fi, not the cable.

**After a Jetson reboot, run `link` first.** Windows does not reliably rejoin
an SSID that vanished mid-connection; `link` reconnects for you and prints
`link down; reconnecting to CrackNet ...`.

One time per PC, already done on this one:

```powershell
.\cracknet.bat join
```

Prove the AP carries it, with the USB route excluded:

```powershell
cmd /c "set JETSON_HOSTS=10.42.0.1 && cracknet check"
```

---

## Give the Jetson internet back

**In AP mode the Jetson has no internet** — the radio does AP *or* client,
never both. Needed before `apt-get`, e.g. installing Gazebo:

```powershell
.\cracknet.bat netmode wifi
```

Back to reachable-without-cables:

```powershell
.\cracknet.bat netmode ap
```

> `netmode` is the one command that cannot safely travel over the link it
> changes. **Plug the USB cable in first.** Without it the launcher warns and
> asks before continuing, and `netmode wifi` then leaves the board reachable
> only by USB or a monitor.

---

## Odd jobs

Label each camera, to write `rig.json`:

```powershell
.\cracknet.bat identify
```

Run one command on the Jetson:

```powershell
.\cracknet.bat shell "nvpmodel -q; free -h"
```

---

## If it will not connect

```powershell
.\cracknet.bat link
```

```powershell
ping 10.42.0.1
```

In order: is the Jetson powered on; does `link` show `CrackNet`; does
`.\cracknet.bat join` fix it; is the USB cable available as a fallback
(`192.168.55.1`).

A non-zero exit with output above it is **not** a connection problem — a failed
preflight exits non-zero on purpose.

Background and the measurements behind all of this:
[docs/network-link.md](docs/network-link.md) ·
[docs/commands.md](docs/commands.md)
