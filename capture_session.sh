#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: ./capture_session.sh [options]

Starts Tanner + Snare stacks, starts tcpdump, and archives run artifacts on shutdown.
The artifact directory is captures/<run-start-timestamp>/.

Options:
  --iface <name>       Capture interface (default: auto-detect tanner_local bridge, fallback: any)
  --pcap-dir <path>    Directory for run artifacts (default: ./captures)
  --filter <bpf>       tcpdump BPF filter (default: tcp)
  --build              Rebuild images before starting services
  --no-down            Keep stacks running on shutdown (default: bring stacks down)
  --dry-run            Print commands without executing
  --strict-firewall    Fail if bridge forwarding rule cannot be verified/inserted
  -h, --help           Show this help

Examples:
  ./capture_session.sh
  ./capture_session.sh --filter 'tcp port 80 or tcp port 8090'
  ./capture_session.sh --build --iface any
USAGE
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TANNER_COMPOSE="${ROOT_DIR}/tanner/docker/docker-compose.yml"
SNARE_COMPOSE="${ROOT_DIR}/snare/docker-compose.yml"
PCAP_DIR="${ROOT_DIR}/captures"
BPF_FILTER="tcp"
BUILD=0
DRY_RUN=0
FIREWALL_STRICT=0
STOP_STACKS=1
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
    --no-down)
      STOP_STACKS=0
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

export_first_existing_file() {
  local container="$1"
  local output_path="$2"
  shift 2
  local candidates=("$@")

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] docker exec ${container} cat <first-existing-candidate> > ${output_path}"
    return 0
  fi

  if ! docker ps --format '{{.Names}}' | grep -Fxq "$container"; then
    echo "[warn] Container '$container' is not running; skipping ${output_path}"
    return 0
  fi

  local source_path
  local temp_path
  temp_path="${output_path}.tmp"

  for source_path in "${candidates[@]}"; do
    if docker exec "$container" sh -lc "test -f '$source_path'" >/dev/null 2>&1; then
      if docker exec "$container" sh -lc "cat '$source_path'" > "$temp_path"; then
        mv "$temp_path" "$output_path"
        echo "[ok] Saved ${container}:${source_path} -> ${output_path}"
        return 0
      fi
      rm -f "$temp_path"
      echo "[warn] Found ${container}:${source_path} but failed to export it"
      return 0
    fi
  done

  rm -f "$temp_path"
  echo "[warn] No matching file found in '$container' for ${output_path}"
}

copy_container_stdout() {
  local container="$1"
  local output_path="$2"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] docker logs --timestamps ${container} > ${output_path}"
    return 0
  fi

  if ! docker ps -a --format '{{.Names}}' | grep -Fxq "$container"; then
    echo "[warn] Container '${container}' does not exist; skipping docker logs"
    return 0
  fi

  if docker logs --timestamps "$container" >"$output_path" 2>&1; then
    echo "[ok] Saved docker logs for '$container'"
  else
    echo "[warn] Failed to save docker logs for '$container'"
  fi
}

record_run_metadata_start() {
  local metadata_path="$1"
  cat > "$metadata_path" <<META
run_start_utc=${RUN_START_UTC}
run_start_epoch=${RUN_START_EPOCH}
interface=${IFACE}
bpf_filter=${BPF_FILTER}
root_dir=${ROOT_DIR}
git_commit=$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)
snare_compose=${SNARE_COMPOSE}
tanner_compose=${TANNER_COMPOSE}
META
}

append_run_metadata_end() {
  local metadata_path="$1"
  {
    echo "run_end_utc=${RUN_END_UTC}"
    echo "run_end_epoch=${RUN_END_EPOCH}"
    echo "run_duration_seconds=$((RUN_END_EPOCH - RUN_START_EPOCH))"
    echo "tcpdump_pid=${TCPDUMP_PID:-unknown}"
    echo "pcap_path=${PCAP_PATH}"
  } >> "$metadata_path"
}

cleanup() {
  local trap_exit_code=$?
  if [[ "${CLEANED_UP:-0}" -eq 1 ]]; then
    return "$trap_exit_code"
  fi
  CLEANED_UP=1

  set +e
  RUN_END_UTC="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  RUN_END_EPOCH="$(date +%s)"

  echo "[info] Shutdown requested; archiving run artifacts to ${RUN_DIR}"

  if [[ -n "${TCPDUMP_PID:-}" ]]; then
    if kill -0 "$TCPDUMP_PID" >/dev/null 2>&1; then
      kill "$TCPDUMP_PID" >/dev/null 2>&1
      wait "$TCPDUMP_PID" >/dev/null 2>&1
      echo "[ok] tcpdump stopped (pid=${TCPDUMP_PID})"
    else
      echo "[warn] tcpdump process already exited (pid=${TCPDUMP_PID})"
    fi
  fi

  export_first_existing_file "snare" "${RUN_DIR}/snare.log" "/opt/snare/snare.log" "/tmp/snare.log"
  export_first_existing_file "snare" "${RUN_DIR}/snare.err" "/opt/snare/snare.err" "/tmp/snare.err"
  export_first_existing_file "tanner" "${RUN_DIR}/tanner.log" "/tmp/tanner/tanner.log" "/opt/tanner/tanner.log" "/var/log/tanner/tanner.log"
  export_first_existing_file "tanner" "${RUN_DIR}/tanner.err" "/tmp/tanner/tanner.err" "/opt/tanner/tanner.err" "/var/log/tanner/tanner.err"

  copy_container_stdout "snare" "${RUN_DIR}/snare.docker.log"
  copy_container_stdout "tanner" "${RUN_DIR}/tanner.docker.log"
  copy_container_stdout "tanner_api" "${RUN_DIR}/tanner_api.docker.log"
  copy_container_stdout "tanner_web" "${RUN_DIR}/tanner_web.docker.log"
  copy_container_stdout "tanner_redis" "${RUN_DIR}/tanner_redis.docker.log"
  copy_container_stdout "tanner_phpox" "${RUN_DIR}/tanner_phpox.docker.log"

  append_run_metadata_end "${RUN_INFO_PATH}"

  if [[ "$STOP_STACKS" -eq 1 ]]; then
    echo "[info] Stopping stacks"
    docker-compose -f "$SNARE_COMPOSE" down || echo "[warn] Failed to stop snare stack"
    docker-compose -f "$TANNER_COMPOSE" down || echo "[warn] Failed to stop tanner stack"
  else
    echo "[info] Leaving stacks running (--no-down)"
  fi

  echo "[ok] Run artifacts archived in ${RUN_DIR}"
  return "$trap_exit_code"
}

require_file "$TANNER_COMPOSE"
require_file "$SNARE_COMPOSE"

RUN_START_UTC="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
RUN_START_EPOCH="$(date +%s)"
RUN_START_TS="$(date +'%Y%m%d_%H%M%S')"
RUN_DIR="${PCAP_DIR}/${RUN_START_TS}"
PCAP_PATH="${RUN_DIR}/capture.pcap"
TCPDUMP_LOG_PATH="${RUN_DIR}/tcpdump.log"
RUN_INFO_PATH="${RUN_DIR}/run_info.txt"
CLEANED_UP=0

mkdir -p "$RUN_DIR"

if [[ -z "$IFACE" ]]; then
  IFACE="$(detect_iface)"
fi

record_run_metadata_start "$RUN_INFO_PATH"

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
  echo "[dry-run] tcpdump -i $IFACE -nn -s 0 -U -w $PCAP_PATH $BPF_FILTER"
  echo "[dry-run] run directory: $RUN_DIR"
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

trap cleanup EXIT INT TERM

echo "[info] Starting tcpdump on interface '$IFACE'"
"${TCPDUMP_PREFIX[@]}" tcpdump -i "$IFACE" -nn -s 0 -U -w "$PCAP_PATH" "$BPF_FILTER" >"$TCPDUMP_LOG_PATH" 2>&1 &
TCPDUMP_PID=$!

echo "[ok] tcpdump started (pid=${TCPDUMP_PID})"
echo "[ok] Run directory: ${RUN_DIR}"
echo "[info] Press Ctrl+C to stop capture and archive logs"

wait "$TCPDUMP_PID"
