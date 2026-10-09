#!/usr/bin/env bash
# netlink.sh - choose how the Jetson is reachable over the network.
#
#   ./netlink.sh ap          host a private Wi-Fi AP, reachable at 10.42.0.1
#   ./netlink.sh wifi        rejoin the infrastructure Wi-Fi (gets internet)
#   ./netlink.sh status      which mode is live, and every usable address
#   ./netlink.sh psk         print the AP password (for the Windows profile)
#   ./netlink.sh boot ap     pick the mode to come up in after a reboot
#   ./netlink.sh boot wifi
#
# WHY AN AP AT ALL
#
# The obvious answer is "put both machines on the campus Wi-Fi and SSH across
# it", and it does not work here. Two independent walls:
#
#   * IOT-PROJ has client isolation. Both machines associate and both reach the
#     internet, but they cannot exchange a single packet with each other. The
#     gateway answers ping; the other station never does.
#   * The wired jack has link (carrier=1) but DHCP times out, because the port
#     only serves registered MACs. The Windows box is registered; the Jetson
#     is not.
#
# Neither is something this script can fix from the Jetson's side. So the
# Jetson stops being a client and becomes the AP instead: it owns the subnet,
# so nothing can isolate anything, and no DHCP server has to approve us. The
# Windows machine joins and keeps its own internet on its Ethernet port.
#
# THE COST, STATED PLAINLY
#
# The rtl88x2ce radio advertises no valid interface combinations, so it does
# AP *or* client, never both. In ap mode the Jetson has no internet. That is
# fine for running and viewing CrackNet, and wrong for apt-get, so `wifi`
# switches back and `boot` decides which one survives a reboot.
set -u

SSID="${CRACKNET_AP_SSID:-CrackNet}"
AP_CON="cracknet-ap"
AP_ADDR="10.42.0.1/24"
BAND="${CRACKNET_AP_BAND:-bg}"      # bg = 2.4 GHz, a = 5 GHz
IFACE="$(nmcli -t -f DEVICE,TYPE dev status | awk -F: '$2=="wifi"{print $1; exit}')"

if [ -n "${NO_COLOR:-}" ] || [ ! -t 1 ]; then
    red() { printf '%s\n' "$*"; }
    grn() { printf '%s\n' "$*"; }
    ylw() { printf '%s\n' "$*"; }
else
    red() { printf '\033[31m%s\033[0m\n' "$*"; }
    grn() { printf '\033[32m%s\033[0m\n' "$*"; }
    ylw() { printf '\033[33m%s\033[0m\n' "$*"; }
fi
info() { printf '  %-22s %s\n' "$1" "$2"; }

need_root() {
    if [ "$(id -u)" -ne 0 ]; then
        red "needs root: sudo ./netlink.sh $*"
        exit 1
    fi
}

# The infrastructure connections, i.e. every wifi profile that is not our AP.
# Discovered rather than hardcoded to IOT-PROJ so a different site needs no
# edit here.
sta_cons() {
    nmcli -t -f NAME,TYPE connection show |
        awk -F: -v ap="$AP_CON" '$2=="802-11-wireless" && $1!=ap {print $1}'
}

# -------------------------------------------------------------- create the AP
ensure_ap() {
    local psk="${1:-}"

    if nmcli -t -f NAME connection show | grep -qx "$AP_CON"; then
        [ -n "$psk" ] && nmcli connection modify "$AP_CON" wifi-sec.psk "$psk"
        return 0
    fi

    # A fresh random password beats a memorable one baked into a tracked file.
    # Nobody types it: `netlink.sh psk` hands it to the Windows side, which
    # installs a wlan profile from it.
    if [ -z "$psk" ]; then
        psk=$(tr -dc 'a-z2-9' < /dev/urandom | head -c 12)
        ylw "  generated a new AP password (netlink.sh psk prints it)"
    fi

    nmcli connection add type wifi ifname "$IFACE" con-name "$AP_CON" \
        autoconnect no ssid "$SSID" >/dev/null

    # ipv4.method shared is what brings up dnsmasq, so the Windows box gets an
    # address without anything else being configured. The address is pinned
    # rather than left to NM's default so the launcher can rely on 10.42.0.1.
    nmcli connection modify "$AP_CON" \
        802-11-wireless.mode ap \
        802-11-wireless.band "$BAND" \
        802-11-wireless.powersave 2 \
        wifi-sec.key-mgmt wpa-psk \
        wifi-sec.proto rsn \
        wifi-sec.pairwise ccmp \
        wifi-sec.group ccmp \
        wifi-sec.psk "$psk" \
        ipv4.method shared \
        ipv4.addresses "$AP_ADDR" \
        ipv6.method ignore \
        connection.autoconnect-priority 20
}

# powersave on the radio shows up as periodic multi-hundred-ms stalls in an
# MJPEG stream, which reads as "the camera is laggy" rather than as a network
# setting. Off on the AP, off on the client.
no_powersave() {
    local c
    for c in $(sta_cons); do
        nmcli connection modify "$c" 802-11-wireless.powersave 2 2>/dev/null || true
    done
}

# ------------------------------------------------------------------- ap / wifi
mode_ap() {
    need_root ap
    ensure_ap "${1:-}"
    no_powersave

    # Autoconnect decides what happens after a reboot; setting it here as well
    # as in `boot` means the mode you switch to is the mode that comes back.
    local c
    for c in $(sta_cons); do
        nmcli connection modify "$c" connection.autoconnect no
    done
    nmcli connection modify "$AP_CON" connection.autoconnect yes

    echo "bringing up the AP on $IFACE ..."
    if ! nmcli -w 40 connection up "$AP_CON" >/dev/null 2>&1; then
        red "  AP failed to start"
        ylw "  restoring the client Wi-Fi autoconnect, so the board is not"
        ylw "  left unreachable after a reboot"
        for c in $(sta_cons); do
            nmcli connection modify "$c" connection.autoconnect yes
        done
        nmcli connection modify "$AP_CON" connection.autoconnect no
        journalctl -u NetworkManager -n 15 --no-pager | sed 's/^/    /'
        return 1
    fi
    grn "  AP up"
    echo
    status
}

mode_wifi() {
    need_root wifi
    no_powersave

    local c first=""
    for c in $(sta_cons); do
        nmcli connection modify "$c" connection.autoconnect yes
        [ -z "$first" ] && first="$c"
    done
    nmcli connection show "$AP_CON" >/dev/null 2>&1 &&
        nmcli connection modify "$AP_CON" connection.autoconnect no

    if [ -z "$first" ]; then
        red "no client Wi-Fi profile exists; create one with:"
        red "  sudo nmcli device wifi connect <SSID> password <PASS>"
        return 1
    fi

    nmcli connection down "$AP_CON" >/dev/null 2>&1 || true
    echo "rejoining $first ..."
    if nmcli -w 45 connection up "$first" >/dev/null 2>&1; then
        grn "  joined $first"
    else
        red "  could not join $first"
    fi
    echo
    status
}

# ---------------------------------------------------------------------- status
status() {
    local active
    active=$(nmcli -t -f NAME,DEVICE connection show --active |
             awk -F: -v i="$IFACE" '$2==i{print $1; exit}')

    if [ "$active" = "$AP_CON" ]; then
        grn "mode: ap - the Jetson is the access point"
        info "ssid" "$SSID"
        info "join at" "${AP_ADDR%/*}"
        info "clients" "$(ip neigh show dev "$IFACE" 2>/dev/null | grep -c .)"
    elif [ -n "$active" ]; then
        grn "mode: wifi - client of $active"
        ylw "  reachable from the PC only if that AP permits station-to-station"
        ylw "  traffic. IOT-PROJ does not."
    else
        ylw "mode: none - the radio has no active connection"
    fi

    echo
    echo "addresses"
    ip -4 -brief addr | grep -vE '^(lo|docker)' |
        awk '{split($3,a,"/"); if (a[1]!="") printf "  %-22s %s\n", $1, a[1]}'

    echo
    echo "after a reboot"
    nmcli -t -f NAME,TYPE,AUTOCONNECT connection show |
        awk -F: '$2=="802-11-wireless"{printf "  %-22s autoconnect %s\n", $1, $3}'
    return 0
}

psk() {
    need_root psk
    # -s is the only way to get the secret out; without it NM prints the
    # placeholder <hidden> and the Windows profile silently gets that string.
    nmcli -s -g 802-11-wireless-security.psk connection show "$AP_CON" 2>/dev/null ||
        { red "no $AP_CON connection yet - run: sudo ./netlink.sh ap"; return 1; }
}

boot() {
    need_root boot "$@"
    local want="${1:-}"
    local c
    case "$want" in
        ap)
            nmcli connection modify "$AP_CON" connection.autoconnect yes
            for c in $(sta_cons); do
                nmcli connection modify "$c" connection.autoconnect no
            done
            grn "will come up as an AP" ;;
        wifi)
            nmcli connection modify "$AP_CON" connection.autoconnect no 2>/dev/null || true
            for c in $(sta_cons); do
                nmcli connection modify "$c" connection.autoconnect yes
            done
            grn "will rejoin the client Wi-Fi" ;;
        *)  red "usage: netlink.sh boot ap|wifi"; return 2 ;;
    esac
}

case "${1:-status}" in
    ap)     shift; mode_ap "${1:-}" ;;
    wifi)   mode_wifi ;;
    status) status ;;
    psk)    psk ;;
    boot)   shift; boot "$@" ;;
    *)      sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 1 ;;
esac
