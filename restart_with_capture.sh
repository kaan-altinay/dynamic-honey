#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./restart_with_capture.sh [options]

Restarts Tanner + Snare docker-compose stacks and starts tcpdump capture
with a timestamped output filename.

Options:
  --iface <name>       Capture interface (default: auto-detect tanner_local bridge, fallback: any)
  --pcap-dir <path>    Directory for pcap output (default: ./captures)
  --filter <bpf>       tcpdump BPF filter (default: tcp)
  --build              Rebuild images before starting services
  --dry-run            Print commands without executing
  --strict-firewall    Fail if bridge forwarding rule cannot be verified/inserted
  -h, --help           Show this help

Examples:
  ./restart_with_capture.sh
  ./restart_with_capture.sh --iface any --filter 'tcp port 80 or tcp port 8090'
  ./restart_with_capture.sh --build
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TANNER_COMPOSE="${ROOT_DIR}/tanner/docker/docker-compose.yml"
SNARE_COMPOSE="${ROOT_DIR}/snare/docker-compose.yml"
PCAP_DIR="${ROOT_DIR}/captures"
BPF_FILTER="tcp"
BUILD=0
DRY_RUN=0
FIREWALL_STRICT=0
IFACE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --iface)
      IFACE="${2:-}"
      shift 2
      ;;
    --pcap-dir)
      PCAP_DIR="${2:-}"
      shift 2
      ;;
    --filter)
      BPF_FILTER="${2:-}"
      shift 2
      ;;
    --build)
      BUILD=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --strict-firewall)
      FIREWALL_STRICT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

require_file() {
  local file="$1"
  if [[ ! -f "$file" ]]; then
    echo "Required file not found: $file" >&2
    exit 1
  fi
}

run_cmd() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[dry-run]'
    for arg in "$@"; do
      printf ' %q' "$arg"
    done
    printf '\n'
  else
    "$@"
  fi
}

detect_iface() {
  local network_id
  network_id="$(docker network inspect -f '{{.Id}}' tanner_local 2>/dev/null || true)"
  if [[ -z "$network_id" ]]; then
    echo "any"
    return
  fi

  local bridge_iface="br-${network_id:0:12}"
  if ip link show "$bridge_iface" >/dev/null 2>&1; then
    echo "$bridge_iface"
  else
    echo "any"
  fi
}

ensure_bridge_forward_rule() {
  local network_id
  network_id="$(docker network inspect -f '{{.Id}}' tanner_local 2>/dev/null || true)"
  if [[ -z "$network_id" ]]; then
    if [[ "$FIREWALL_STRICT" -eq 1 ]]; then
      echo "[error] Docker network 'tanner_local' not found; cannot ensure forwarding rule" >&2
      exit 1
    fi
    echo "[warn] Docker network 'tanner_local' not found; skipping forwarding rule check"
    return
  fi

  local bridge_iface="br-${network_id:0:12}"
  local sudo_prefix=()

  if [[ "$EUID" -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
      sudo_prefix=(sudo)
    else
      if [[ "$FIREWALL_STRICT" -eq 1 ]]; then
        echo "[error] Root privileges are required to manage iptables and sudo is unavailable" >&2
        exit 1
      fi
      echo "[warn] Cannot ensure forwarding rule without root or sudo; continuing"
      return
    fi
  fi

  if ! command -v iptables >/dev/null 2>&1; then
    if [[ "$FIREWALL_STRICT" -eq 1 ]]; then
      echo "[error] iptables command not found; cannot ensure forwarding rule" >&2
      exit 1
    fi
    echo "[warn] iptables command not found; skipping forwarding rule check"
    return
  fi

  if "${sudo_prefix[@]}" iptables -C DOCKER-USER -i "$bridge_iface" -o "$bridge_iface" -j ACCEPT >/dev/null 2>&1; then
    echo "[info] Docker bridge forwarding rule already present for '$bridge_iface'"
    return
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[dry-run]'
    for arg in "${sudo_prefix[@]}" iptables -I DOCKER-USER 1 -i "$bridge_iface" -o "$bridge_iface" -j ACCEPT; do
      printf ' %q' "$arg"
    done
    printf '\n'
    echo "[info] Docker bridge forwarding rule would be added for '$bridge_iface'"
    return
  fi

  if "${sudo_prefix[@]}" iptables -I DOCKER-USER 1 -i "$bridge_iface" -o "$bridge_iface" -j ACCEPT; then
    echo "[ok] Added Docker bridge forwarding rule for '$bridge_iface'"
    return
  fi

  if [[ "$FIREWALL_STRICT" -eq 1 ]]; then
    echo "[error] Failed to add forwarding rule for '$bridge_iface'" >&2
    exit 1
  fi
  echo "[warn] Failed to add forwarding rule for '$bridge_iface'; continuing"
}

require_file "$TANNER_COMPOSE"
require_file "$SNARE_COMPOSE"

if [[ -z "$IFACE" ]]; then
  IFACE="$(detect_iface)"
fi

mkdir -p "$PCAP_DIR"
TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
PCAP_PATH="${PCAP_DIR}/snare_tanner_${TIMESTAMP}.pcap"
LOG_PATH="${PCAP_DIR}/snare_tanner_${TIMESTAMP}.tcpdump.log"

UP_TANNER_CMD=(docker-compose -f "$TANNER_COMPOSE" up -d)
UP_SNARE_CMD=(docker-compose -f "$SNARE_COMPOSE" up -d)
if [[ "$BUILD" -eq 1 ]]; then
  UP_TANNER_CMD+=(--build)
  UP_SNARE_CMD+=(--build)
fi

echo "[info] Restarting stacks"
run_cmd docker-compose -f "$SNARE_COMPOSE" down
run_cmd docker-compose -f "$TANNER_COMPOSE" down
ensure_bridge_forward_rule
run_cmd "${UP_TANNER_CMD[@]}"
run_cmd "${UP_SNARE_CMD[@]}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] tcpdump -i $IFACE -nn -s 0 -w $PCAP_PATH $BPF_FILTER"
  echo "[dry-run] pcap: $PCAP_PATH"
  exit 0
fi

if ! command -v tcpdump >/dev/null 2>&1; then
  echo "tcpdump not found in PATH" >&2
  exit 1
fi

TCPDUMP_PREFIX=()
if [[ "$EUID" -ne 0 ]]; then
  if ! command -v sudo >/dev/null 2>&1; then
    echo "tcpdump capture requires root privileges. Run as root or install sudo." >&2
    exit 1
  fi
  TCPDUMP_PREFIX=(sudo)
fi

echo "[info] Starting tcpdump on interface '$IFACE'"
nohup "${TCPDUMP_PREFIX[@]}" tcpdump -i "$IFACE" -nn -s 0 -U -w "$PCAP_PATH" "$BPF_FILTER" >"$LOG_PATH" &
TCPDUMP_PID=$!

echo "[ok] tcpdump started"
echo "[ok] PID: $TCPDUMP_PID"
echo "[ok] PCAP: $PCAP_PATH"
echo "[ok] LOG:  $LOG_PATH"
