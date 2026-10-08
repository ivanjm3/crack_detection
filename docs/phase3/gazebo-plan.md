# Phase 3 — Gazebo simulation of the tunnel inspection robot

Build plan. Written to be executed from a cold start after this conversation is
compacted.

**Budget: 3 hours.** Everything below is time-boxed, with a fallback at each
step that cannot overrun.

---

## 1. What is being simulated

A **four-wheeled robot driving through a road tunnel** (the kind cars and trucks
use), inspecting the tunnel lining for cracks.

Three cameras cover the tunnel cross-section:

```
            TOP  ↑ (crown)
             ┌───────────┐
            ╱             ╲
   LEFT ←  │   [robot]     │  → RIGHT
   (wall)  │    ▄▄▄▄▄      │    (wall)
           └───────────────┘
                 road
```

| Camera | Points | Sees |
|---|---|---|
| `left` | perpendicular, to the left | left tunnel wall |
| `top` | straight up | tunnel crown |
| `right` | perpendicular, to the right | right tunnel wall |

This is why the roles are named left/top/right rather than by position in a row.

**Motion model: stop-and-scan.** The robot drives to a position, stops, captures
(~6.5 s for all three cameras), then advances. This matches the real pipeline,
which depends on the scene being static — see
[../phase2/01-camera-weaving.md](../phase2/01-camera-weaving.md) §2.

---

## 2. The finding that shapes the design

**Read this before building anything.**

Resolution is set by standoff. With a C920 (70.42° horizontal FOV, 1920 px):

| Standoff | Swath | mm/px | Smallest crack (2 px) |
|---|---|---|---|
| 0.40 m | 0.56 m | 0.29 | **0.59 mm** |
| 0.75 m | 1.06 m | 0.55 | 1.10 mm |
| 1.50 m | 2.12 m | 1.10 | 2.21 mm |
| 3.00 m | 4.23 m | 2.21 | 4.41 mm |
| **4.50 m** (wall of a road tunnel) | 6.35 m | 3.31 | **6.62 mm** |
| **5.50 m** (crown) | 7.76 m | 4.04 | **8.09 mm** |

A road tunnel is roughly 9–10 m wide and 6–7 m high. A robot in the middle of
the carriageway is therefore **4–6 m from the lining**, where a C920 resolves
cracks of about **7–8 mm**.

Tunnel inspection standards care about cracks from roughly **0.2–0.3 mm**. So:

> **A C920 in the middle of a road tunnel cannot see the cracks that matter.**
> It is off by more than an order of magnitude.

This is not a flaw to hide in the simulation. It is the most valuable thing the
simulation can demonstrate, and it is the obvious question an examiner asks.

### The four ways out, and which the sim should show

| Option | Effect | Cost |
|---|---|---|
| **Drive close to one wall** (0.4–0.75 m) | 0.29–0.55 mm/px — meets spec | Only 0.5–1 m of lining per pass; needs many passes, and cannot reach the crown |
| **Telephoto lenses** | Keeps standoff, restores mm/px | Different hardware; much narrower swath per camera |
| **Mast / arm** carrying cameras near the lining | Meets spec, reaches crown | Mechanical complexity |
| **Accept coarse detection** | Finds 7 mm+ structural cracks only | Fails the durability inspection use case |

**Recommended for the sim:** model the full-size tunnel honestly, and give the
robot **two modes** — a centre pass (fast, coarse, 3–4 mm/px) and a wall-hugging
pass (slow, fine, 0.3–0.5 mm/px). The console already computes and displays
mm/px from standoff, so the difference appears on screen with no extra work.

---

## 3. Software choice

| Decision | Choice | Why |
|---|---|---|
| Where it runs | **Jetson** | WSL is not installed on the Windows laptop; installing WSL2 + Ubuntu + Gazebo needs a reboot, admin rights, and more than the whole budget |
| Simulator | **Gazebo Fortress** (`gz-fortress`) | Confirmed available for jammy/arm64. Pairs with Ubuntu 22.04. Garden and Harmonic are also available if Fortress misbehaves |
| ROS | **None** | `ros-humble-desktop` is ~2 GB. At the measured 662 KB/s that is ~50 minutes of a 180-minute budget, for a message bus this build does not need |
| Rendering | **Headless** (`gz sim -s`) | No X session on the Jetson |
| Image egress | **Camera sensor saves frames to disk** | Avoids gz-transport Python bindings entirely. Crude, robust, and fast enough at stop-and-scan rates |

### Streaming the simulation to Windows — solved with what already exists

No VNC, no X11 forwarding, no extra GUI.

1. The sim's three cameras write frames to disk.
2. A `SimRig` class feeds those frames into the **real** detection pipeline, in
   place of the USB cameras.
3. `serve_scan.py` serves them through the **existing dashboard** at
   `http://192.168.55.1:8081/`, which the Windows laptop already opens.
4. A **fourth camera in the world** — a chase camera behind the robot — gives the
   third-person 3D view of the robot in the tunnel, streamed through the same
   MJPEG mechanism.

So the Windows machine sees the robot driving through the tunnel *and* the live
crack detection, over plain HTTP, with no new software on either side.

---

## 4. Build order, time-boxed

### Step 1 — Install and prove rendering (30 min, **hard cutoff 45**)

```bash
sudo apt-get update
sudo apt-get install -y lsb-release wget gnupg
sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/gazebo-stable.list
sudo apt-get update
sudo apt-get install -y gz-fortress
gz sim --versions
```

Then prove headless rendering works before building anything:

```bash
gz sim -s -r -v 3 --iterations 100 shapes.sdf
```

**If rendering fails:** try `--render-engine ogre` (ogre1 instead of ogre2).
**If it still fails:** go to the fallback in §6.

### Step 2 — Tunnel world (30 min)

`sim/tunnel.sdf`

- Arched tunnel ~**9 m wide, 6.5 m high**, ~60 m long. Build the arch from a
  half-cylinder and two vertical walls, or one scaled cylinder.
- Road surface, lane markings for scale and realism.
- **Crack texture on the lining.** Generate procedurally (reuse the approach in
  `inject_crack.py`: meandering, tapered, soft-edged, multiplicative darkening)
  and apply as a material texture.
  - **Texture scale is the critical parameter.** It must be applied so one
    texture pixel is a known number of millimetres on the lining, or the
    measured widths are meaningless.
- Lighting: tunnel lights at intervals. Dim, uneven lighting is realistic and
  will exercise the auto-exposure loop.
- Optional if time allows: a parked truck for scale.

### Step 3 — Robot (25 min)

`sim/robot.sdf`

**The robot is KINEMATIC, not driven.** This is the single most important
scoping decision in the plan.

A diff-drive robot needs wheel friction, inertia tensors, a controller and PID
tuning, and when it misbehaves it slides, tips or refuses to turn — each of
which is an hour of debugging on its own. None of that appears in the output.
What the demo shows is **what the cameras see from a sequence of positions**, and
that is fully determined by the pose.

So the model is `<static>` with four wheels modelled for appearance only, and
the pose is set directly:

```bash
gz service -s /world/tunnel/set_pose --reqtype gz.msgs.Pose   --reptype gz.msgs.Boolean --timeout 300   --req 'name: "inspector", position: {x: 12.0, y: 0, z: 0.35}'
```

Stop-and-scan then becomes exactly what it is physically: set a pose, wait for
the render to settle, capture, advance. No controller, no physics tuning, and
the motion is repeatable to the millimetre — which matters, because a survey
that cannot return to the same position cannot be compared between runs.

If there is time left at the end, swap in `gz-sim-diff-drive-system`. It changes
nothing downstream.

- Chassis box ~**0.6 × 0.4 × 0.25 m**, four wheels, radius 0.1 m (visual only).
- Four cameras, all `<camera>` sensors with:
  - `<horizontal_fov>1.2291</horizontal_fov>` (70.42° in radians, matching the C920)
  - 1920×1080 (drop to 1280×720 if the frame rate is poor)
  - `<save enabled="true"><path>/tmp/simcam/<role></path></save>`

| Name | Pose | Aim |
|---|---|---|
| `left` | left side of chassis | +Y, at the left wall |
| `top` | chassis top | +Z, at the crown |
| `right` | right side | −Y, at the right wall |
| `chase` | behind and above | forward, for the 3D view |

`sim/drive.sh` steps the pose down the tunnel, pausing at each station long
enough for the cameras to write a settled frame.

### Step 4 — Bridge into the real pipeline (25 min)

`jetson/sim_rig.py`

A `SimRig` class with the same interface as `SequentialRig`:

```python
class SimRig:
    def capture(self):   # -> {role: (frame, stats)}
```

It reads the newest image per role from `/tmp/simcam/<role>/`, returning the
same structure the USB rig returns, with `exposure`/`gain` reported as `None`
(the sim has no v4l2 controls).

Then `serve_scan.py --source sim` uses it. **Everything downstream is
unchanged** — tiling, inference, shape filtering, measurement, the console.

### Step 4b — Crack texture (15 min)

Generate the lining texture procedurally, physically scaled, before wiring the
bridge — if the texture is wrong the end-to-end test proves nothing.

### Step 5 — Verify end to end (20 min)

- Robot drives, stops, three frames land on disk
- Pipeline runs, cracks are detected
- Console shows detections and a sane mm/px for the modelled standoff
- Chase camera visible from Windows

### Buffer: 25 min

| Step | Minutes | Running total |
|---|---|---|
| 1. Install + prove headless rendering | 30 | 30 |
| 2. Tunnel world | 30 | 60 |
| 3. Robot + 4 cameras (kinematic) | 25 | 85 |
| 4. SimRig bridge into the pipeline | 25 | 110 |
| 4b. Crack texture | 15 | 125 |
| 5. End-to-end verification | 20 | 145 |
| Buffer | 25 | **170** |

The three hours cover **everything**, not just the install. If any step runs
over its box, cut from the bottom: the chase camera is the first thing to drop,
then the lighting detail, then the second (wall-hugging) pass.

---

## 5. Files to create

```
sim/
  tunnel.sdf          world: tunnel, road, lights, crack-textured lining
  robot.sdf           four-wheel chassis, diff-drive, 4 cameras
  textures/
    lining_crack.png  procedurally generated, physically scaled
  drive.sh            stop-and-scan motion script
jetson/
  sim_rig.py          SimRig - feeds sim frames into the real pipeline
docs/phase3/
  gazebo-plan.md      this file
```

---

## 6. Fallback — if Gazebo will not render on the Jetson

This is the real risk: ogre2 headless rendering on Jetson ARM can fail, and
debugging it could eat the whole budget.

**Hard cutoff: if rendering is not working 45 minutes in, abandon Gazebo.**

Fallback (~45 min, essentially zero risk): a **procedural tunnel renderer** in
OpenCV. Render the tunnel interior from the robot's three camera poses with
perspective projection onto a crack-textured cylinder, advancing the robot along
the tunnel. It feeds the same `SimRig` interface, so steps 4 and 5 are
unchanged, and the console demo is identical.

What is lost: physics, and the word "Gazebo". What is kept: the tunnel, the
three-camera geometry, the standoff/resolution relationship, and a real
end-to-end detection demo.

---

## 7. Decisions still needed

1. ~~Does the assignment require ROS?~~ **Resolved: no.** It is a simulation
   only, with no ROS requirement, so the no-ROS plan above stands and the
   ~50 minutes it would have cost stays in the budget. If ROS is ever needed
   later, `ros_gz_bridge` can publish the existing camera topics without
   changing anything in this build.
2. **Tunnel size.** This plan assumes a full-size road tunnel (9 m × 6.5 m),
   which makes the resolution limit in §2 real and visible. A service tunnel
   (2–3 m) would let the C920 meet spec and remove the finding. The road tunnel
   is the more honest and more interesting choice.
3. **Which camera resolution in the sim** — 1920×1080 matches the hardware but
   renders slowly on a Jetson; 1280×720 is a reasonable compromise.

---

## 8. What to say about it

The simulation demonstrates three things, in order of value:

1. **A resolution budget that real geometry imposes.** Three C920s in the middle
   of a road tunnel resolve ~7 mm cracks, not the 0.3 mm that matters. The sim
   shows the trade between standoff, swath and detectable width, and the
   wall-hugging pass that fixes it.
2. **The three-camera cross-section** covering wall, crown, wall — with
   detection running per-camera, never on a stitched image.
3. **The complete pipeline running on synthetic input**, proving the perception
   stack is independent of where the frames come from.
