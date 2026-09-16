#!/usr/bin/env bash
# cracknet.sh - start / stop / inspect the whole CrackNet system on the Jetson.
#
#   ./cracknet.sh start [serve.py flags...]   bring everything up
#   ./cracknet.sh stop                        shut it down, release the camera
#   ./cracknet.sh restart [flags...]          stop then start
#   ./cracknet.sh status                      what is running, and is it healthy
#   ./cracknet.sh logs                        follow the server log
#   ./cracknet.sh check                       preflight only, change nothing
#
# start performs, in order: preflight checks -> performance mode -> camera
# control lock-down -> server. Anything that needs sudo is best-effort: if
# passwordless sudo is unavailable it warns and carries on rather than hanging
# on a password prompt.
#
# Examples
#   ./cracknet.sh start --rotate 180
#   ./cracknet.sh start --thresh 0.6 --min-area 400
#   ./cracknet.sh restart --no-auto-exposure

set -u

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR" || exit 1

PIDFILE="$DIR/cracknet.pid"
LOG="$DIR/serve.log"
PORT=8080
ENGINE="cracknet_fp16.engine"
CAM=/dev/video0
# Matches the running server but NOT this script's own command line - a looser
# pattern makes pkill kill the shell that invoked it.
PATTERN='python3 serve\.py'

red()  { printf '\033[31m%s\033[0m\n' "$*"; }
grn()  { printf '\033[32m%s\033[0m\n' "$*"; }
ylw()  { printf '\033[33m%s\033[0m\n' "$*"; }
info() { printf '  %-30s %s\n' "$1" "$2"; }

server_pid() {
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        cat "$PIDFILE"; return 0
    fi
    pgrep -f "$PATTERN" | head -1
}

urls() {
    ip -4 -brief addr | grep -vE '^(lo|docker)' |
        awk -v p="$PORT" '{split($3,a,"/"); if (a[1] != "") print "    http://"a[1]":"p"/"}'
}

# ---------------------------------------------------------------- preflight
check() {
    local fail=0
    echo "preflight"

    if python3 -c 'import tensorrt' 2>/dev/null; then
        info "tensorrt" "$(python3 -c 'import tensorrt;print(tensorrt.__version__)' 2>/dev/null)"
    else
        info "tensorrt" "MISSING - run: sudo apt install nvidia-jetpack"; fail=1
    fi

    for m in cv2 pycuda numpy; do
        python3 -c "import $m" 2>/dev/null \
            && info "$m" "ok" \
            || { info "$m" "MISSING"; fail=1; }
    done

    if [ -s "$ENGINE" ]; then
        info "engine" "$ENGINE ($(du -h "$ENGINE" | cut -f1))"
    else
        info "engine" "MISSING - run: bash build_engines.sh"; fail=1
    fi

    if [ -e "$CAM" ]; then
        local fmt
        fmt=$(v4l2-ctl -d "$CAM" --get-fmt-video 2>/dev/null |
              awk -F"'" '/Pixel Format/{print $2}')
        info "camera" "$CAM ${fmt:+($fmt)}"
    else
        info "camera" "NOT PRESENT - plug the C920 into a USB-A port"; fail=1
    fi

    info "power mode" "$(nvpmodel -q 2>/dev/null | head -1 | sed 's/.*: //')"
    return $fail
}

# -------------------------------------------------------------------- start
start() {
    if [ -n "$(server_pid)" ]; then
        ylw "already running (pid $(server_pid)); use restart"
        status
        return 0
    fi

    check || { red "preflight failed - not starting"; return 1; }

    echo
    echo "performance mode"
    if sudo -n true 2>/dev/null; then
        # NB: mode 0 is 15 W on this board. MAXN_SUPER is mode 2.
        sudo -n nvpmodel -m 2 </dev/null >/dev/null 2>&1
        sudo -n jetson_clocks </dev/null >/dev/null 2>&1
        info "nvpmodel" "$(nvpmodel -q 2>/dev/null | head -1 | sed 's/.*: //')"
    else
        ylw "  no passwordless sudo - skipping nvpmodel/jetson_clocks"
        ylw "  run manually: sudo nvpmodel -m 2 && sudo jetson_clocks"
    fi

    echo
    echo "camera"
    bash camera_setup.sh "$CAM" >/dev/null 2>&1
    info "focus/exposure" "locked (live.py re-applies after stream start)"

    echo
    echo "server"
    nohup python3 serve.py --port "$PORT" "$@" > "$LOG" 2>&1 &
    echo $! > "$PIDFILE"

    # Wait for the port rather than sleeping a fixed amount: engine load plus
    # auto-exposure settling takes 10-20 s depending on the scene.
    local i
    for i in $(seq 1 40); do
        if curl -sf -o /dev/null "http://127.0.0.1:$PORT/snapshot.jpg" 2>/dev/null; then
            break
        fi
        if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
            red "  server died during startup:"
            sed 's/^/    /' "$LOG" | tail -15
            rm -f "$PIDFILE"
            return 1
        fi
        sleep 1
    done

    if curl -sf -o /dev/null "http://127.0.0.1:$PORT/snapshot.jpg" 2>/dev/null; then
        grn "  up (pid $(cat "$PIDFILE"))"
        info "auto-exposure" "$(grep -m1 'auto-exposure settled' "$LOG" | sed 's/.*settled: //')"
        echo
        echo "view at:"
        urls
    else
        red "  no frames after 40 s - check: ./cracknet.sh logs"
        return 1
    fi
}

# --------------------------------------------------------------------- stop
stop() {
    local pid
    pid=$(server_pid)
    if [ -z "$pid" ]; then
        ylw "not running"
        rm -f "$PIDFILE"
        return 0
    fi

    kill "$pid" 2>/dev/null
    local i
    for i in $(seq 1 10); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.5
    done
    if kill -0 "$pid" 2>/dev/null; then
        ylw "  did not exit on SIGTERM, sending SIGKILL"
        kill -9 "$pid" 2>/dev/null
        sleep 1
    fi

    pkill -f "$PATTERN" 2>/dev/null     # sweep up any stragglers
    rm -f "$PIDFILE"

    # The camera is the resource worth confirming: a held /dev/video0 blocks
    # every other script with a confusing "can't open camera by index".
    if command -v fuser >/dev/null && fuser "$CAM" >/dev/null 2>&1; then
        ylw "  warning: $CAM still held by pid $(fuser "$CAM" 2>/dev/null)"
    else
        grn "  stopped, camera released"
    fi
}

# ------------------------------------------------------------------- status
status() {
    local pid
    pid=$(server_pid)

    if [ -n "$pid" ]; then
        grn "running (pid $pid)"
        info "uptime" "$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')"
        info "command" "$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-60)"
    else
        ylw "not running"
    fi

    if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
        info "port $PORT" "listening"
    else
        info "port $PORT" "closed"
    fi

    local js
    js=$(curl -sf --max-time 5 "http://127.0.0.1:$PORT/status.json" 2>/dev/null)
    if [ -n "$js" ]; then
        # %-formatting with double quotes only: nested same-type quotes inside
        # an f-string are a syntax error on the Jetson's Python 3.10.
        echo "$js" | python3 -c '
import json, sys
d = json.load(sys.stdin)
def g(k, fmt="%s"):
    v = d.get(k)
    return "-" if v is None else fmt % v
print("  %-30s %s  coverage %.2f%%" % (
      "detection", "CRACK" if d["crack"] else "clear", 100 * d["coverage"]))
print("  %-30s %.1f ms inference, %.1f fps" % (
      "performance", d["infer_ms"], d["fps"]))
print("  %-30s exp=%s gain=%s median=%s clipped=%.1f%%" % (
      "exposure", g("exposure"), g("gain"), g("median", "%.0f"),
      100 * (d.get("clipped") or 0)))
' 2>/dev/null || info "status.json" "$js"
    fi

    [ -n "$pid" ] && { echo; echo "view at:"; urls; }
    return 0
}

# --------------------------------------------------------------------- main
case "${1:-status}" in
    start)   shift; start "$@" ;;
    stop)    stop ;;
    restart) shift; stop; echo; start "$@" ;;
    status)  status ;;
    check)   check ;;
    logs)    tail -f "$LOG" ;;
    *)
        sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
        exit 1
        ;;
esac
