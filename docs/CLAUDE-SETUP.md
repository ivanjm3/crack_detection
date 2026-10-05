# Setting up on another machine

How to pick this project up on a different computer — the repo, the Jetson
connection, and the Claude Code conversation history.

---

## 1. What this project is

CrackNet: a crack-segmentation model deployed on a **Jetson Orin Nano** with Logitech
C920 cameras. The model is trained and frozen; the work here is the **perception
pipeline** around it — camera control, exposure, geometry, mask post-processing,
and a network viewer.

Start with these, in order:

| Read | For |
| --- | --- |
| `docs/perception.html` | How the whole pipeline works, stage by stage, with diagrams. Open it in a browser |
| `docs/commands.md` | Driving the Jetson from PowerShell: connect, start/stop, deploy, diagnose |
| `docs/multicam-plan.md` | The current work: three-camera stationary scanning |
| `docs/WORKLOG-2026-09-16.md` | What was tried, what failed, and why — read before re-trying anything |
| `jetson/README.md` | File index and the seven places `setup.md.txt` is wrong for this board |

---

## 2. Clone the repo

```bash
git clone https://github.com/ivanjm3/crack_detection.git
cd crack_detection
```

### What is NOT in the repo

Deliberately excluded by `.gitignore`, because they are large or machine-specific:

| Missing | Why | How to get it |
| --- | --- | --- |
| `crack_outputs/*.pt`, `*.onnx` | ~159 MB of weights | Copy from the lab machine or Google Drive |
| `cracknet_*.engine` | Tied to one board **and** one TensorRT version | Rebuild on the device: `bash build_engines.sh` |
| `val_samples/`, `*.mp4` | Captured data | Regenerate: `python3 capture_samples.py` |

**Never copy a TensorRT engine between machines.** It will either refuse to
deserialise or behave incorrectly. Always rebuild.

---

## 3. Prerequisites on the new machine

| Need | For |
| --- | --- |
| **Git** | the repo. On Windows it often isn't on `PATH` — see §6 |
| **Python 3** + `paramiko` | `tools/jssh.py`, the SSH helper (`pip install paramiko`) |
| **OpenSSH** (`ssh`, `scp`) | manual access. Built into Windows 10/11 at `C:\Windows\System32\OpenSSH\` |
| **Claude Code** | optional, but see §5 for carrying the conversation over |

Nothing else is needed on the host. CUDA, TensorRT and OpenCV live on the Jetson,
not here.

---

## 4. Connecting to the Jetson

### 4.1 It has to be physically reachable

This is the part that does not travel.

| Address | What it is | Reachable from |
| --- | --- | --- |
| `192.168.55.1` | USB device-mode link | **only the machine the USB cable is plugged into** |
| `172.16.61.x` | campus Wi-Fi (`IOT-PROJ`), DHCP | only on that network — **not from home** |

The Jetson is a physical box. From a different location you can read, plan and write
code, but you **cannot** run anything that touches the camera, the engine or the
board. Plan accordingly: write code remotely, test on-site.

The Wi-Fi address is a DHCP lease and **changes** — it moved from `.175` to `.180`
within one day. Never hardcode it; read it from `./cracknet.sh status`.

Its ethernet port does not work: the link comes up at 1000 Mbps but DHCP never
answers, because the MAC (`3c:6d:66:bf:3a:b2`) is not registered on the campus
network. Wi-Fi is the working path.

### 4.2 Credentials

Not stored in this repo. Set them in the environment:

```powershell
# PowerShell
$env:JETSON_HOST = "192.168.55.1"
$env:JETSON_USER = "sarah"
$env:JETSON_PASS = "<ask>"
```

```bash
# bash
export JETSON_HOST=192.168.55.1 JETSON_USER=sarah JETSON_PASS=<ask>
```

Or create `tools/.jetson.env` (gitignored):

```
JETSON_HOST=192.168.55.1
JETSON_USER=sarah
JETSON_PASS=<ask>
```

### 4.3 Driving it

Interactively:

```bash
ssh sarah@192.168.55.1
cd ~/cracknet && ./cracknet.sh status
```

Scripted, which is what Claude Code uses — password prompts can't be answered by a
non-interactive tool, so everything goes through `tools/jssh.py`:

```bash
python tools/jssh.py "cd ~/cracknet && ./cracknet.sh status"
python tools/jssh.py --sudo "nvpmodel -m 2"
python tools/jssh.py --file local_script.sh
python tools/jssh.py --put jetson/live.py /home/sarah/cracknet/live.py
```

> **Git Bash on Windows:** POSIX paths in arguments are rewritten to Windows paths
> before Python sees them, so a remote path like `/home/sarah/x.py` fails with
> "No such file". Prefix with `MSYS_NO_PATHCONV=1`.

### 4.4 Deploying code

The repo is the source of truth. Edit here, push to the device, never the reverse:

```bash
python tools/jssh.py --put jetson/live.py /home/sarah/cracknet/live.py
python tools/jssh.py "cd ~/cracknet && ./cracknet.sh restart"
```

`docs/commands.md` §4 has a one-liner that checksums both sides to confirm they match.

---

## 5. Carrying a Claude Code conversation across machines

Two separate things: the **project** (git, §2) and the **conversation** (local files).

### 5.1 What to copy

From the old machine:

```
C:\Users\<user>\.claude\projects\C--Users-<user>-Desktop-rp-proj\
    <session-uuid>.jsonl     the conversation transcript
    memory\                  4 small files - project knowledge that auto-loads
```

`memory\` is the one people forget. It is what makes a brand-new session already know
how to reach the Jetson, which parts of `setup.md.txt` are wrong, and why the detector
fires on people. It is small and worth copying even if you skip the transcript.

### 5.2 Getting the folder name right

The folder under `.claude\projects\` encodes the project's **path**, so it differs
per machine. Don't work it out by hand:

1. Clone the repo on the new machine.
2. `cd` into it, run `claude`, exit immediately.
3. A new folder appears under `~\.claude\projects\` — that's the one.
4. Copy the `.jsonl` and `memory\` into it.

### 5.3 Resuming

```bash
claude --resume <session-uuid>     # continue that exact conversation
claude -c                          # continue the most recent one here
claude -r                          # interactive picker
claude --resume <uuid> --fork-session   # new session ID, original left intact
```

### 5.4 Or just start fresh — often better

A long transcript can be several megabytes, and resuming replays all of it, so the new
session begins with most of its context already spent on finished work.

The docs in §1 plus the auto-loading `memory\` carry nearly all the useful state. A
fresh session in the repo directory, pointed at `docs/multicam-plan.md`, starts clean
and knows almost as much.

### 5.5 Related commands, and which direction they go

| Command | Moves |
| --- | --- |
| `/remote-control` (`/rc`) | this **local** session → phone/browser. Keeps running locally; the original session must stay open |
| `/teleport` (`/tp`), `claude --teleport` | a **cloud** session → your terminal. The wrong direction for a local session |
| `claude --resume <id>` | this conversation → a new process, anywhere the files are |

---

## 6. Windows notes

Git is frequently installed but not on `PATH`:

```powershell
$env:Path += ";C:\Users\<user>\AppData\Local\Programs\Git\cmd"
```

PowerShell 5.1 has **no `&&` operator**. Chain with `;`, or `cmd1; if ($?) { cmd2 }`.
(`&&` inside a quoted *remote* command is fine — that runs in bash on the Jetson.)

`ssh -t` is needed for anything that prompts for sudo or that you will Ctrl-C.

---

## 7. Where the work stands

**Done and verified on the device:** FP16 engine at 8.77 ms, TensorRT/ONNX parity at
0.0552 % mask disagreement, camera controls locked, auto-exposure with a simulated
test suite at 7/7, shape filtering removing 96.2 % of false-positive coverage, and an
MJPEG viewer on port 8080.

**Current work:** three-camera stationary scanning — `docs/multicam-plan.md`.

**Next action:** plug three C920s into Type-A ports on the board and run

```bash
python tools/jssh.py "cd ~/cracknet && python3 multicam_probe.py"
python tools/jssh.py "cd ~/cracknet && python3 check_fov_parity.py"
```

**Open decision, needed before the rig is built:** the smallest crack width the survey
must detect. It fixes the standoff and therefore the swath — 0.3 mm needs ~0.20 m
standoff and gives a 0.71 m swath; 0.6 mm allows 0.40 m and 1.43 m. See
`docs/multicam-plan.md` §4A.4.

**Known gaps, stated honestly:**

- Threshold tuning (`setup.md` §7) has never been done on real pavement. Every number
  outside the parity check was measured on out-of-domain indoor scenes.
- The model cannot say "this is not pavement". Shape filtering mitigates it; it is not
  a fix. See `docs/perception.html` §5.
- The viewer is unauthenticated and bound to `0.0.0.0`.
- There is no boot service; the viewer runs under `nohup` and dies on reboot.

---

## 8. Credentials hygiene

The Jetson password is weak and has been shared in plain text in chat. Before this
goes anywhere public, change it (`passwd` on the device) and update
`tools/.jetson.env`. The same applies to any GitHub password that has been typed into
a chat window — rotate it, and use a personal access token or SSH key for git, since
[GitHub removed password authentication for git operations in 2021](https://github.blog/2020-12-15-token-authentication-requirements-for-git-operations/).
