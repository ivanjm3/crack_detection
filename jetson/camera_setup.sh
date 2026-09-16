#!/usr/bin/env bash
# Lock down C920 autofocus / auto-exposure (setup.md 4.2).
# Control names differ across kernels, so each setting is tried both ways.
DEV="${1:-/dev/video0}"

set_ctrl() {
    # try each name=value pair until one sticks
    for kv in "$@"; do
        if v4l2-ctl -d "$DEV" -c "$kv" 2>/dev/null; then
            echo "  ok: $kv"
            return 0
        fi
    done
    echo "  FAILED (none applied): $*" >&2
    return 1
}

echo "configuring $DEV"
set_ctrl focus_automatic_continuous=0 focus_auto=0
set_ctrl focus_absolute=30
set_ctrl auto_exposure=1 exposure_auto=1
# must be off before exposure is pinned: with it on the driver re-extends
# exposure_time_absolute behind our back (observed drifting 156 -> 312)
set_ctrl exposure_dynamic_framerate=0
set_ctrl exposure_time_absolute=156 exposure_absolute=156
set_ctrl power_line_frequency=1

echo "current values:"
v4l2-ctl -d "$DEV" --list-ctrls | grep -iE 'focus|exposure|power_line' || true
