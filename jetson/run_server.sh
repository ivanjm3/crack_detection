#!/usr/bin/env bash
# Start (or restart) the MJPEG viewer on the Jetson.
#   ./run_server.sh              defaults
#   ./run_server.sh --rotate 180 pass any serve.py flag through
#
# View at http://<jetson-ip>:8080/ - `ip -4 -brief addr` lists the addresses.
cd "$(dirname "$0")" || exit 1

# NB: the pattern must not match this script's own command line, or pkill
# kills the shell running it.
pkill -f 'python3 serve\.py' 2>/dev/null
sleep 2

nohup python3 serve.py --port 8080 "$@" > serve.log 2>&1 &
sleep 15

if pgrep -f 'python3 serve\.py' >/dev/null; then
    echo "running: $(pgrep -af 'python3 serve\.py' | head -1)"
    echo -n "status: "; curl -s http://127.0.0.1:8080/status.json; echo
    ip -4 -brief addr | grep -vE '^lo' | awk '{split($3,a,"/"); print "  http://"a[1]":8080/"}'
else
    echo "FAILED to start; log follows:" >&2
    cat serve.log >&2
    exit 1
fi
