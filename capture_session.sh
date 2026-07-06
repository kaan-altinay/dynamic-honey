#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./capture_session.sh start --run-name <name> (--range <start-end> | --range-start <start>) [options]
  ./capture_session.sh stop --run-name <name> [options]
  ./capture_session.sh status --run-name <name> [options]
  ./capture_session.sh list [options]
  ./capture_session.sh stop-legacy [options]

Commands:
  start         Start one dynamic-honey instance for a contiguous 4-IP public range.
  stop          Stop a named instance and archive docker logs into its run directory.
  status        Show status for a named instance.
  list          List known run directories and their recorded metadata.
  stop-legacy   Stop the currently running legacy snare/tanner stack started from the base compose files.

Options:
  --run-name <name>       Required for start/stop/status. Used as the capture directory name under captures_new/.
                          Example: default_first_run, cache_first_run.
  --mode <name>           default | agentic | agentic-v2 | spoof200 | current | cache (default: agentic)
                          default   -> same source tree, but with GENERATOR.backend forced to none via temp config.
                          agentic/current/cache -> use the repo config as-is.
                          agentic-v2 -> same as agentic, but with GENERATOR.enable_scripted_flows forced true
                                        in the per-run Tanner config copy.
                          spoof200  -> Tanner is not started at all; Snare answers every request with an empty
                                       HTTP 200 OK directly, without contacting Tanner.
  --range <start-end>     Public IP host suffix range, must be a contiguous block of 4 aligned on 16,20,24,28.
                          Example: 16-19
  --range-start <start>   Equivalent shorthand for a 4-IP block. Example: --range-start 16 means 16-19.
  --captures-dir <path>   Artifact root (default: ./captures_new)
  --iface <name>          Capture interface (default: enp6s20)
  --page-url <host>       Snare page-dir / PAGE_URL (default: example.com)
  --forward-ports <list>  Comma-separated host ports forwarded to Snare :80
                          (default: 80,443,8080,8081,8000,8888,8443,9200,8983,5984,2375,6443,5000,8500,5701,8848,5985,5986,17778)
  --cliproxy-dir <path>   Path to cli-proxy-api working directory (default: /home/kaan/cliproxyapi)
  --cliproxy-cmd <cmd>    Command to launch backend from cliproxy-dir (default: ./cli-proxy-api)
  --no-cliproxy-auto      Do not auto-start cli-proxy-api for agentic mode
  --build                 Rebuild images before starting services
  --dry-run               Print commands without executing
  -h, --help              Show this help

Notes:
  - Start one instance per 4-IP range. Up to four concurrent runs can cover 16-19, 20-23, 24-27, and 28-31.
  - This script uses the same source tree for every run. Separate run directories keep captures, runtime files,
    and snare state isolated.
  - The "default" mode disables the agentic backend via a temporary Tanner config overlay. It does not restore a
    pristine upstream source tree; for that you would need a separate clean checkout or image.
  - The "agentic-v2" mode keeps the normal agentic stack, but flips GENERATOR.enable_scripted_flows=true
    only in that run's generated Tanner config, so V2 overrides apply without touching the shared base config.
  - The "spoof200" mode runs Snare standalone: no Tanner/Tanner-web/Tanner-api/Redis/phpox containers are started,
    and Snare never makes a network call to Tanner for any request.
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_TANNER_COMPOSE="${ROOT_DIR}/tanner/docker/docker-compose.yml"
BASE_SNARE_COMPOSE="${ROOT_DIR}/snare/docker-compose.yml"
BASE_TANNER_CONFIG="${ROOT_DIR}/tanner/tanner/data/config.yaml"
CAPTURES_DIR="${ROOT_DIR}/captures_new"
SNARE_TEMPLATE_DIR="/home/kaan/snare-data/snare"
PUBLIC_IP_PREFIX="145.220.178"
DEFAULT_PAGE_URL="example.com"
DEFAULT_IFACE="enp6s20"
DEFAULT_FORWARD_PORTS="80,443,8080,8081,8000,8888,8443,9200,8983,5984,2375,6443,5000,8500,5701,8848,5985,5986,17778"
DEFAULT_CLIPROXY_DIR="/home/kaan/cliproxyapi"
DEFAULT_CLIPROXY_CMD="./cli-proxy-api"

COMMAND="${1:-}"
if [[ -n "$COMMAND" ]]; then
  shift
fi

RUN_NAME=""
MODE="agentic"
RANGE=""
RANGE_START=""
IFACE="${DEFAULT_IFACE}"
PAGE_URL="${DEFAULT_PAGE_URL}"
BUILD=0
DRY_RUN=0
FORWARD_PORTS="${DEFAULT_FORWARD_PORTS}"
CLIPROXY_DIR="${DEFAULT_CLIPROXY_DIR}"
CLIPROXY_CMD="${DEFAULT_CLIPROXY_CMD}"
CLIPROXY_AUTO=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name)
      RUN_NAME="${2:-}"
      shift 2
      ;;
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --range)
      RANGE="${2:-}"
      shift 2
      ;;
    --range-start)
      RANGE_START="${2:-}"
      shift 2
      ;;
    --captures-dir)
      CAPTURES_DIR="${2:-}"
      shift 2
      ;;
    --iface)
      IFACE="${2:-}"
      shift 2
      ;;
    --page-url)
      PAGE_URL="${2:-}"
      shift 2
      ;;
    --forward-ports)
      FORWARD_PORTS="${2:-}"
      shift 2
      ;;
    --cliproxy-dir)
      CLIPROXY_DIR="${2:-}"
      shift 2
      ;;
    --cliproxy-cmd)
      CLIPROXY_CMD="${2:-}"
      shift 2
      ;;
    --no-cliproxy-auto)
      CLIPROXY_AUTO=0
      shift
      ;;
    --build)
      BUILD=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
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

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Required command not found: $cmd" >&2
    exit 1
  fi
}

require_sudo_if_needed() {
  if [[ "$EUID" -ne 0 ]] && ! command -v sudo >/dev/null 2>&1; then
    echo "This action requires root privileges or sudo." >&2
    exit 1
  fi
}

compose_bin() {
  if command -v docker-compose >/dev/null 2>&1; then
    echo "docker-compose"
    return
  fi
  if docker compose version >/dev/null 2>&1; then
    echo "docker compose"
    return
  fi
  echo "Docker Compose is not available" >&2
  exit 1
}

COMPOSE_BIN_STR="$(compose_bin)"
read -r -a COMPOSE_BIN <<<"${COMPOSE_BIN_STR}"

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

run_compose() {
  local -a args=("${COMPOSE_BIN[@]}" "$@")
  run_cmd "${args[@]}"
}

need_sudo_for_signal() {
  local pid="$1"
  local owner
  owner="$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
  if [[ -z "$owner" || "$owner" == "$USER" ]]; then
    return 1
  fi
  return 0
}

kill_pid_if_running() {
  local pid="$1"
  if [[ -z "$pid" ]]; then
    return 0
  fi
  if ! kill -0 "$pid" >/dev/null 2>&1; then
    return 0
  fi
  if need_sudo_for_signal "$pid"; then
    require_sudo_if_needed
    run_cmd sudo kill "$pid"
  else
    run_cmd kill "$pid"
  fi
}

slugify() {
  local value="$1"
  value="$(echo "$value" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//')"
  printf '%s' "$value"
}

normalize_mode() {
  case "$1" in
    default) echo "default" ;;
    agentic|current|cache) echo "agentic" ;;
    agentic-v2) echo "agentic-v2" ;;
    spoof200) echo "spoof200" ;;
    *)
      echo "Unsupported mode: $1" >&2
      exit 1
      ;;
  esac
}

mode_uses_agentic_backend() {
  [[ "$1" == "agentic" || "$1" == "agentic-v2" ]]
}

parse_range() {
  local start end
  if [[ -n "$RANGE" && -n "$RANGE_START" ]]; then
    echo "Specify either --range or --range-start, not both." >&2
    exit 1
  fi
  if [[ -n "$RANGE" ]]; then
    if [[ ! "$RANGE" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      echo "Invalid --range format. Expected start-end, for example 16-19." >&2
      exit 1
    fi
    start="${BASH_REMATCH[1]}"
    end="${BASH_REMATCH[2]}"
  elif [[ -n "$RANGE_START" ]]; then
    start="$RANGE_START"
    end="$((RANGE_START + 3))"
  else
    echo "A 4-IP public range is required. Use --range 16-19 or --range-start 16." >&2
    exit 1
  fi

  if (( start < 16 || end > 31 || end - start != 3 )); then
    echo "Range must stay within 16-31 and contain exactly 4 IPs." >&2
    exit 1
  fi
  if (( (start - 16) % 4 != 0 )); then
    echo "Range must start on one of 16, 20, 24, or 28." >&2
    exit 1
  fi

  RANGE_START="$start"
  RANGE_END="$end"
  GROUP_INDEX="$(((start - 16) / 4))"
}

build_ip_list() {
  IP_LIST=()
  local suffix
  for ((suffix = RANGE_START; suffix <= RANGE_END; suffix++)); do
    IP_LIST+=("${PUBLIC_IP_PREFIX}.${suffix}")
  done
}


build_forward_port_list() {
  if [[ -z "${FORWARD_PORTS}" ]]; then
    echo "--forward-ports cannot be empty." >&2
    exit 1
  fi

  FORWARD_PORT_LIST=()
  local raw_port trimmed
  local -A seen=()
  IFS=',' read -r -a raw_ports <<<"${FORWARD_PORTS}"
  for raw_port in "${raw_ports[@]}"; do
    trimmed="${raw_port//[[:space:]]/}"
    if [[ -z "${trimmed}" ]]; then
      continue
    fi
    if [[ ! "${trimmed}" =~ ^[0-9]+$ ]]; then
      echo "Invalid port in --forward-ports: ${trimmed}" >&2
      exit 1
    fi
    if (( trimmed < 1 || trimmed > 65535 )); then
      echo "Port out of range in --forward-ports: ${trimmed}" >&2
      exit 1
    fi
    if [[ -n "${seen[${trimmed}]:-}" ]]; then
      continue
    fi
    seen["${trimmed}"]=1
    FORWARD_PORT_LIST+=("${trimmed}")
  done

  if (( ${#FORWARD_PORT_LIST[@]} == 0 )); then
    echo "No valid ports resolved from --forward-ports." >&2
    exit 1
  fi
}

build_tcpdump_filter() {
  local host_parts=()
  local ip
  for ip in "${IP_LIST[@]}"; do
    host_parts+=("host ${ip}")
  done
  local joined
  joined="$(printf ' or %s' "${host_parts[@]}")"
  joined="${joined:4}"
  TCPDUMP_FILTER="tcp and (${joined})"
}

init_run_layout() {
  RUN_DIR="${CAPTURES_DIR}/${RUN_NAME}"
  RUNTIME_DIR="${RUN_DIR}/runtime"
  STATE_DIR="${RUN_DIR}/snare_state"

  RUN_INFO_PATH="${RUN_DIR}/run_info.env"
  PCAP_PATH="${RUN_DIR}/capture.pcap"
  TCPDUMP_LOG_PATH="${RUN_DIR}/tcpdump.log"
  TANNER_COMPOSE_PATH="${RUNTIME_DIR}/tanner.compose.yml"
  SNARE_COMPOSE_PATH="${RUNTIME_DIR}/snare.compose.yml"
  TANNER_CONFIG_PATH="${RUNTIME_DIR}/tanner.config.yaml"

  INSTANCE_SLUG="$(slugify "$RUN_NAME")"
  PROJECT_NAME="dh-${INSTANCE_SLUG}"
  NETWORK_NAME="${PROJECT_NAME}-net"
  WEB_PORT="$((8091 + GROUP_INDEX))"

  SNARE_CONTAINER="${PROJECT_NAME}-snare"
  TANNER_CONTAINER="${PROJECT_NAME}-tanner"
  TANNER_API_CONTAINER="${PROJECT_NAME}-tanner-api"
  TANNER_WEB_CONTAINER="${PROJECT_NAME}-tanner-web"
  TANNER_REDIS_CONTAINER="${PROJECT_NAME}-tanner-redis"
  TANNER_PHPOX_CONTAINER="${PROJECT_NAME}-tanner-phpox"
}

select_capture_paths() {
  PCAP_PATH="${RUN_DIR}/capture.pcap"
  TCPDUMP_LOG_PATH="${RUN_DIR}/tcpdump.log"
  if [[ ! -e "$RUN_DIR" || ! -e "$PCAP_PATH" ]]; then
    return 0
  fi
  local idx=2
  while :; do
    local candidate_pcap="${RUN_DIR}/capture_${idx}.pcap"
    if [[ ! -e "$candidate_pcap" ]]; then
      PCAP_PATH="$candidate_pcap"
      TCPDUMP_LOG_PATH="${RUN_DIR}/tcpdump_${idx}.log"
      return 0
    fi
    idx=$((idx + 1))
  done
}

prepare_run_dirs() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi
  mkdir -p "$RUN_DIR" "$RUNTIME_DIR" "$STATE_DIR"
}


ensure_page_dir_writable() {
  local target_dir="$1"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] ensure writable page dir ${target_dir}"
    return 0
  fi
  if [[ ! -d "$target_dir" ]]; then
    return 0
  fi

  if [[ -w "$target_dir" ]]; then
    chmod u+rwx "$target_dir" || true
  fi

  local test_file="${target_dir}/.write-test.$$"
  if touch "$test_file" >/dev/null 2>&1; then
    rm -f "$test_file"
    return 0
  fi

  if command -v sudo >/dev/null 2>&1; then
    sudo chmod -R a+rwX "$target_dir"
    return 0
  fi

  echo "Page directory is not writable and sudo is unavailable: ${target_dir}" >&2
  exit 1
}
seed_snare_state() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] prepare snare state in ${STATE_DIR}"
    return 0
  fi
  mkdir -p "$STATE_DIR"
  if [[ -d "$SNARE_TEMPLATE_DIR" ]]; then
    if [[ ! -e "$STATE_DIR/pages" ]]; then
      mkdir -p "$STATE_DIR/pages"
    fi
    if [[ ! -d "$STATE_DIR/pages/${PAGE_URL}" && -d "$SNARE_TEMPLATE_DIR/pages/${PAGE_URL}" ]]; then
      run_cmd rsync -a "$SNARE_TEMPLATE_DIR/pages/${PAGE_URL}" "$STATE_DIR/pages/"
    fi
    if [[ ! -f "$STATE_DIR/seedfile.txt" && -f "$SNARE_TEMPLATE_DIR/seedfile.txt" ]]; then
      run_cmd cp "$SNARE_TEMPLATE_DIR/seedfile.txt" "$STATE_DIR/seedfile.txt"
    fi
  fi

  # Snare drops privileges to nobody; ensure the page directory remains writable
  # without failing on existing generated artifacts owned by another uid.
  ensure_page_dir_writable "$STATE_DIR/pages/${PAGE_URL}"
}

write_tanner_config() {
  local normalized_mode="$1"
  local temp_mode="$normalized_mode"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] write Tanner config overlay -> ${TANNER_CONFIG_PATH} (mode=${temp_mode})"
    return 0
  fi
  python3 - "$BASE_TANNER_CONFIG" "$TANNER_CONFIG_PATH" "$temp_mode" <<'PY'
from pathlib import Path
import re
import sys
src, dst, mode = sys.argv[1:4]
text = Path(src).read_text()
if mode == "default":
    text = re.sub(r'(^\s*backend:\s*).*$','\\1none', text, flags=re.M)
    for emulator_key in ("lfi", "cmd_exec", "template_injection"):
        text = re.sub(rf'(^\s*{re.escape(emulator_key)}:\s*).*$','\\1False', text, flags=re.M)
elif mode == "agentic-v2":
    text = re.sub(r'(^\s*enable_scripted_flows:\s*).*$','\\1true', text, flags=re.M)
text = re.sub(r'(^\s*log_debug:\s*).*$','\\1/var/log/tanner/tanner.log', text, flags=re.M)
text = re.sub(r'(^\s*log_err:\s*).*$','\\1/var/log/tanner/tanner.err', text, flags=re.M)
text = re.sub(r'(^\s*REDIS:\s*\n(?:.*\n)*?\s*host:\s*).*$','\\1tanner_redis', text, flags=re.M)
Path(dst).write_text(text)
PY
}

write_tanner_compose() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] write Tanner compose -> ${TANNER_COMPOSE_PATH}"
    return 0
  fi
  cat > "$TANNER_COMPOSE_PATH" <<EOF
version: '2.3'
services:
  tanner_redis:
    build: '${ROOT_DIR}/tanner/docker/redis'
    image: tanner-redis-local
    container_name: ${TANNER_REDIS_CONTAINER}
    restart: always
    stop_signal: SIGKILL
    tty: true
    networks:
      - local
    read_only: true
    tmpfs:
      - /data:uid=65534,gid=65534

  tanner_phpox:
    build: '${ROOT_DIR}/tanner/docker/phpox'
    image: tanner-phpox-local
    container_name: ${TANNER_PHPOX_CONTAINER}
    restart: always
    stop_signal: SIGKILL
    tty: true
    networks:
      - local
    read_only: true
    tmpfs: /tmp

  tanner_api:
    build:
      context: '${ROOT_DIR}/tanner'
      dockerfile: docker/tanner/Dockerfile.local
    image: tanner-local:patched
    container_name: ${TANNER_API_CONTAINER}
    restart: always
    stop_signal: SIGKILL
    tty: true
    networks:
      - local
    read_only: true
    tmpfs:
      - /tmp/tanner:uid=65534,gid=65534
      - /var/log/tanner:uid=65534,gid=65534
    environment:
      OPENAI_BASE_URL: "${OPENAI_BASE_URL:-}"
      OPENAI_API_BASE: "${OPENAI_API_BASE:-}"
      OPENAI_API_KEY: "${OPENAI_API_KEY:-}"
    extra_hosts:
      - 'host.docker.internal:host-gateway'
    command: ["/opt/tanner/tanner-env/bin/tannerapi", "--config", "/opt/tanner/runtime-config/config.yaml"]
    volumes:
      - '${TANNER_CONFIG_PATH}:/opt/tanner/runtime-config/config.yaml:ro'
    depends_on:
      - tanner_redis

  tanner_web:
    build:
      context: '${ROOT_DIR}/tanner'
      dockerfile: docker/tanner/Dockerfile.local
    image: tanner-local:patched
    container_name: ${TANNER_WEB_CONTAINER}
    restart: always
    stop_signal: SIGKILL
    tty: true
    networks:
      - local
    read_only: true
    tmpfs:
      - /tmp/tanner:uid=65534,gid=65534
      - /var/log/tanner:uid=65534,gid=65534
    environment:
      OPENAI_BASE_URL: "${OPENAI_BASE_URL:-}"
      OPENAI_API_BASE: "${OPENAI_API_BASE:-}"
      OPENAI_API_KEY: "${OPENAI_API_KEY:-}"
    extra_hosts:
      - 'host.docker.internal:host-gateway'
    ports:
      - '127.0.0.1:${WEB_PORT}:8091'
    command: ["/opt/tanner/tanner-env/bin/tannerweb", "--config", "/opt/tanner/runtime-config/config.yaml"]
    volumes:
      - '${TANNER_CONFIG_PATH}:/opt/tanner/runtime-config/config.yaml:ro'
    depends_on:
      - tanner_api
      - tanner_redis

  tanner:
    build:
      context: '${ROOT_DIR}/tanner'
      dockerfile: docker/tanner/Dockerfile.local
    image: tanner-local:patched
    container_name: ${TANNER_CONTAINER}
    restart: always
    stop_signal: SIGKILL
    tty: true
    networks:
      - local
    read_only: true
    tmpfs:
      - /tmp/tanner:uid=65534,gid=65534
      - /var/log/tanner:uid=65534,gid=65534
      - /opt/tanner/files:uid=65534,gid=65534
    environment:
      OPENAI_BASE_URL: "${OPENAI_BASE_URL:-}"
      OPENAI_API_BASE: "${OPENAI_API_BASE:-}"
      OPENAI_API_KEY: "${OPENAI_API_KEY:-}"
    extra_hosts:
      - 'host.docker.internal:host-gateway'
    command: ["/opt/tanner/tanner-env/bin/tanner", "--config", "/opt/tanner/runtime-config/config.yaml"]
    volumes:
      - '${TANNER_CONFIG_PATH}:/opt/tanner/runtime-config/config.yaml:ro'
      - '/var/run/docker.sock:/var/run/docker.sock'
    depends_on:
      - tanner_api
      - tanner_web
      - tanner_phpox

networks:
  local:
    name: ${NETWORK_NAME}
EOF
}

write_snare_compose() {
  local ports_block=""
  local ip host_port
  for ip in "${IP_LIST[@]}"; do
    for host_port in "${FORWARD_PORT_LIST[@]}"; do
      ports_block+="      - '${ip}:${host_port}:80'"$'\n'
    done
  done

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] write Snare compose -> ${SNARE_COMPOSE_PATH}"
    return 0
  fi

  local environment_block="      - TANNER=tanner
      - PAGE_URL=${PAGE_URL}
      - PORT=80
      - PYTHONPATH=/"
  if [[ "${MODE}" == "spoof200" ]]; then
    environment_block+=$'\n      - SPOOF_200=true'
  fi

  # spoof200 never starts Tanner, so Snare's compose project owns the
  # network outright instead of attaching to one Tanner created.
  local network_block
  if [[ "${MODE}" == "spoof200" ]]; then
    network_block="networks:
  local:
    name: ${NETWORK_NAME}"
  else
    network_block="networks:
  local:
    external: true
    name: ${NETWORK_NAME}"
  fi

  cat > "$SNARE_COMPOSE_PATH" <<EOF
version: '2.3'
services:
  snare:
    build: '${ROOT_DIR}/snare'
    image: snare-snare-local
    container_name: ${SNARE_CONTAINER}
    restart: always
    stop_signal: SIGKILL
    tty: true
    networks:
      - local
    ports:
${ports_block%$'\n'}
    environment:
${environment_block}
    volumes:
      - '${STATE_DIR}:/opt/snare'
      - '${ROOT_DIR}/snare/snare:/snare:ro'

${network_block}
EOF
}


ensure_cliproxy_for_agentic() {
  if ! mode_uses_agentic_backend "${MODE}" || [[ "${CLIPROXY_AUTO}" -ne 1 ]]; then
    return 0
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] ensure cli-proxy-api via '${CLIPROXY_CMD}' in ${CLIPROXY_DIR}"
    return 0
  fi

  if [[ ! -d "$CLIPROXY_DIR" ]]; then
    echo "cli-proxy directory not found: $CLIPROXY_DIR" >&2
    exit 1
  fi

  local existing_pid
  existing_pid="$(pgrep -f '[c]li-proxy-api' | head -n 1 || true)"
  if [[ -n "$existing_pid" ]]; then
    echo "[info] cli-proxy-api already running (pid: ${existing_pid})"
    echo "cliproxy_pid=${existing_pid}" >> "$RUN_INFO_PATH"
    echo "cliproxy_started_by_run=0" >> "$RUN_INFO_PATH"
    return 0
  fi

  local launch_cmd
  printf -v launch_cmd 'cd %q && exec %q' "$CLIPROXY_DIR" "$CLIPROXY_CMD"

  local cliproxy_log="${RUN_DIR}/cliproxy.log"
  nohup bash -lc "$launch_cmd" >"$cliproxy_log" 2>&1 &
  local cliproxy_pid=$!
  sleep 1

  if ! kill -0 "$cliproxy_pid" >/dev/null 2>&1; then
    echo "Failed to start cli-proxy-api. Inspect log: ${cliproxy_log}" >&2
    exit 1
  fi

  echo "[ok] cli-proxy-api started (pid: ${cliproxy_pid})"
  echo "cliproxy_pid=${cliproxy_pid}" >> "$RUN_INFO_PATH"
  echo "cliproxy_started_by_run=1" >> "$RUN_INFO_PATH"
}

verify_llm_backend_from_tanner_container() {
  if ! mode_uses_agentic_backend "${MODE}"; then
    return 0
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] verify Tanner container can reach ${OPENAI_BASE_URL:-<unset>}"
    return 0
  fi

  local backend_url="${OPENAI_BASE_URL:-}"
  local backend_key="${OPENAI_API_KEY:-}"
  if [[ -z "$backend_url" ]]; then
    echo "OPENAI_BASE_URL is not set for ${MODE} mode" >&2
    exit 1
  fi

  echo "[info] Verifying Tanner container can reach LLM backend at ${backend_url}"
  if ! docker exec -i "$TANNER_CONTAINER" python3 - "$backend_url" "$backend_key" <<'PY'
import json
import sys
import urllib.request

backend_url = sys.argv[1].rstrip("/")
backend_key = sys.argv[2]
request = urllib.request.Request(
    backend_url + "/models",
    headers={"Authorization": "Bearer " + backend_key} if backend_key else {},
)
with urllib.request.urlopen(request, timeout=10) as response:
    payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("unexpected non-object response from /models")
    payload.get("data", [])
PY
  then
    echo "Tanner container cannot reach the configured LLM backend (${backend_url}); refusing to keep the dynamic stack online." >&2
    run_compose -p "$PROJECT_NAME" -f "$TANNER_COMPOSE_PATH" down --remove-orphans || true
    run_compose -p "$PROJECT_NAME" -f "$SNARE_COMPOSE_PATH" down --remove-orphans || true
    exit 1
  fi
  echo "[ok] Tanner container can reach the configured LLM backend"
}

allow_cliproxy_from_tanner_network() {
  if ! mode_uses_agentic_backend "${MODE}"; then
    return 0
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] allow ${NETWORK_NAME} subnet(s) to reach cli-proxy-api on tcp/8317"
    return 0
  fi

  require_cmd iptables
  require_sudo_if_needed

  local -a prefix=()
  if [[ "$EUID" -ne 0 ]]; then
    prefix=(sudo)
  fi

  local subnet
  local inspected=0
  while IFS= read -r subnet; do
    [[ -z "$subnet" ]] && continue
    inspected=1
    if "${prefix[@]}" iptables -C INPUT -s "$subnet" -p tcp --dport 8317 -j ACCEPT >/dev/null 2>&1; then
      echo "[info] cli-proxy-api firewall rule already allows ${subnet} -> tcp/8317"
      continue
    fi
    "${prefix[@]}" iptables -I INPUT 1 -s "$subnet" -p tcp --dport 8317 -j ACCEPT
    echo "[ok] Added cli-proxy-api firewall allow rule for ${subnet} -> tcp/8317"
  done < <(docker network inspect "$NETWORK_NAME" --format '{{range .IPAM.Config}}{{println .Subnet}}{{end}}')

  if [[ "$inspected" -eq 0 ]]; then
    echo "Could not determine subnet(s) for Docker network ${NETWORK_NAME}" >&2
    exit 1
  fi
}

write_run_info() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] write run metadata -> ${RUN_INFO_PATH}"
    return 0
  fi
  {
    printf 'run_name=%q\n' "$RUN_NAME"
    printf 'mode=%q\n' "$MODE"
    printf 'project_name=%q\n' "$PROJECT_NAME"
    printf 'network_name=%q\n' "$NETWORK_NAME"
    printf 'range_start=%q\n' "$RANGE_START"
    printf 'range_end=%q\n' "$RANGE_END"
    printf 'ip_list=%q\n' "${IP_LIST[*]}"
    printf 'iface=%q\n' "$IFACE"
    printf 'page_url=%q\n' "$PAGE_URL"
    printf 'forward_ports=%q\n' "${FORWARD_PORT_LIST[*]}"
    printf 'web_port=%q\n' "$WEB_PORT"
    printf 'tanner_compose=%q\n' "$TANNER_COMPOSE_PATH"
    printf 'snare_compose=%q\n' "$SNARE_COMPOSE_PATH"
    printf 'tanner_config=%q\n' "$TANNER_CONFIG_PATH"
    printf 'state_dir=%q\n' "$STATE_DIR"
    printf 'pcap_path=%q\n' "$PCAP_PATH"
    printf 'tcpdump_log_path=%q\n' "$TCPDUMP_LOG_PATH"
    printf 'snare_container=%q\n' "$SNARE_CONTAINER"
    printf 'tanner_container=%q\n' "$TANNER_CONTAINER"
    printf 'tanner_api_container=%q\n' "$TANNER_API_CONTAINER"
    printf 'tanner_web_container=%q\n' "$TANNER_WEB_CONTAINER"
    printf 'tanner_redis_container=%q\n' "$TANNER_REDIS_CONTAINER"
    printf 'tanner_phpox_container=%q\n' "$TANNER_PHPOX_CONTAINER"
    printf 'cliproxy_dir=%q\n' "$CLIPROXY_DIR"
    printf 'cliproxy_cmd=%q\n' "$CLIPROXY_CMD"
    printf 'cliproxy_auto=%q\n' "$CLIPROXY_AUTO"
  } > "$RUN_INFO_PATH"
}

start_tcpdump() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] tcpdump -i ${IFACE} -nn -s 0 -U -w ${PCAP_PATH} ${TCPDUMP_FILTER}"
    return 0
  fi
  require_cmd tcpdump

  local -a prefix=()
  if [[ "$EUID" -ne 0 ]]; then
    require_sudo_if_needed
    prefix=(sudo)
  fi

  nohup "${prefix[@]}" tcpdump -i "$IFACE" -nn -s 0 -U -w "$PCAP_PATH" "$TCPDUMP_FILTER" >"$TCPDUMP_LOG_PATH" 2>&1 &
  TCPDUMP_PID=$!
  echo "tcpdump_pid=${TCPDUMP_PID}" >> "$RUN_INFO_PATH"
}

load_run_info() {
  RUN_DIR="${CAPTURES_DIR}/${RUN_NAME}"
  RUN_INFO_PATH="${RUN_DIR}/run_info.env"
  if [[ ! -f "$RUN_INFO_PATH" ]]; then
    echo "Run metadata not found: $RUN_INFO_PATH" >&2
    exit 1
  fi

  local line key raw_value value
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" ]] && continue
    key="${line%%=*}"
    raw_value="${line#*=}"
    printf -v value '%b' "${raw_value//\\ / }"
    case "$key" in
      run_name|mode|project_name|network_name|range_start|range_end|ip_list|iface|page_url|forward_ports|web_port|tanner_compose|snare_compose|tanner_config|state_dir|pcap_path|tcpdump_log_path|snare_container|tanner_container|tanner_api_container|tanner_web_container|tanner_redis_container|tanner_phpox_container|tcpdump_pid|cliproxy_pid|cliproxy_started_by_run|cliproxy_dir|cliproxy_cmd|cliproxy_auto)
        printf -v "$key" '%s' "$value"
        ;;
    esac
  done < "$RUN_INFO_PATH"
}

write_container_logs() {
  local name="$1"
  local output_path="$2"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] docker logs --timestamps ${name} > ${output_path}"
    return 0
  fi
  if docker ps -a --format '{{.Names}}' | grep -Fxq "$name"; then
    docker logs --timestamps "$name" >"$output_path" 2>&1 || true
  fi
}

start_run() {
  MODE="$(normalize_mode "$MODE")"
  if [[ -z "$RUN_NAME" ]]; then
    echo "--run-name is required for start." >&2
    exit 1
  fi
  require_file "$BASE_TANNER_COMPOSE"
  require_file "$BASE_SNARE_COMPOSE"
  require_file "$BASE_TANNER_CONFIG"
  require_cmd docker
  require_cmd python3
  require_cmd rsync

  parse_range
  build_ip_list
  build_forward_port_list
  build_tcpdump_filter
  init_run_layout
  select_capture_paths

  if [[ -e "$RUN_DIR" ]]; then
    echo "[info] Reusing existing run directory: $RUN_DIR"
    echo "[info] New capture artifact will be: ${PCAP_PATH}"
  fi

  prepare_run_dirs
  seed_snare_state

  if mode_uses_agentic_backend "${MODE}"; then
    export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://host.docker.internal:8317/v1}"
    export OPENAI_API_BASE="${OPENAI_API_BASE:-http://host.docker.internal:8317/v1}"
    export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-mor5R6MlcggVCit9qS3XzjqjW4Egc9PCOyZuWZYy1qUrf}"
  fi

  if [[ "${MODE}" != "spoof200" ]]; then
    write_tanner_config "$MODE"
    write_tanner_compose
  fi
  write_snare_compose
  write_run_info

  echo "[info] Starting run '${RUN_NAME}' (${MODE}) for ${PUBLIC_IP_PREFIX}.${RANGE_START}-${RANGE_END}"
  echo "[info] Capture dir: ${RUN_DIR}"
  echo "[info] Snare state: ${STATE_DIR}"
  echo "[info] Forwarded host ports -> snare:80: ${FORWARD_PORT_LIST[*]}"
  if [[ "${MODE}" == "spoof200" ]]; then
    echo "[info] Tanner: not started (spoof200 mode answers every request with a bare 200 OK)"
  else
    echo "[info] Tanner web: http://127.0.0.1:${WEB_PORT}"
  fi
  echo "[info] tcpdump filter: ${TCPDUMP_FILTER}"

  ensure_cliproxy_for_agentic


  if [[ "${MODE}" != "spoof200" ]]; then
    run_compose -p "$PROJECT_NAME" -f "$TANNER_COMPOSE_PATH" down --remove-orphans
  fi
  run_compose -p "$PROJECT_NAME" -f "$SNARE_COMPOSE_PATH" down --remove-orphans

  if [[ "${MODE}" != "spoof200" ]]; then
    if [[ "$BUILD" -eq 1 ]]; then
      run_compose -p "$PROJECT_NAME" -f "$TANNER_COMPOSE_PATH" up -d --build
    else
      run_compose -p "$PROJECT_NAME" -f "$TANNER_COMPOSE_PATH" up -d
    fi
    allow_cliproxy_from_tanner_network
    verify_llm_backend_from_tanner_container
  fi

  if [[ "$BUILD" -eq 1 ]]; then
    run_compose -p "$PROJECT_NAME" -f "$SNARE_COMPOSE_PATH" up -d --build
  else
    run_compose -p "$PROJECT_NAME" -f "$SNARE_COMPOSE_PATH" up -d
  fi

  start_tcpdump

  echo "[ok] Run started"
  echo "[ok] Mode: ${MODE}"
  echo "[ok] IPs: ${IP_LIST[*]}"
  echo "[ok] PCAP: ${PCAP_PATH}"
  if [[ -n "${TCPDUMP_PID:-}" ]]; then
    echo "[ok] tcpdump pid: ${TCPDUMP_PID}"
  fi
}

stop_run() {
  if [[ -z "$RUN_NAME" ]]; then
    echo "--run-name is required for stop." >&2
    exit 1
  fi
  load_run_info

  echo "[info] Stopping run '${run_name}'"
  kill_pid_if_running "${tcpdump_pid:-}"

  write_container_logs "$snare_container" "${RUN_DIR}/snare.docker.log"
  if [[ "${mode}" != "spoof200" ]]; then
    write_container_logs "$tanner_container" "${RUN_DIR}/tanner.docker.log"
    write_container_logs "$tanner_api_container" "${RUN_DIR}/tanner_api.docker.log"
    write_container_logs "$tanner_web_container" "${RUN_DIR}/tanner_web.docker.log"
    write_container_logs "$tanner_redis_container" "${RUN_DIR}/tanner_redis.docker.log"
    write_container_logs "$tanner_phpox_container" "${RUN_DIR}/tanner_phpox.docker.log"
  fi

  run_compose -p "$project_name" -f "$snare_compose" down --remove-orphans
  if [[ "${mode}" != "spoof200" ]]; then
    run_compose -p "$project_name" -f "$tanner_compose" down --remove-orphans
  fi

  echo "[ok] Run stopped: ${run_name}"
}

status_run() {
  if [[ -z "$RUN_NAME" ]]; then
    echo "--run-name is required for status." >&2
    exit 1
  fi
  load_run_info

  echo "run_name=${run_name}"
  echo "mode=${mode}"
  echo "ips=${ip_list}"
  echo "iface=${iface}"
  echo "pcap_path=${pcap_path}"
  echo "web_port=${web_port}"
  echo "forward_ports=${forward_ports:-80}"
  if [[ -n "${cliproxy_pid:-}" ]] && kill -0 "${cliproxy_pid}" >/dev/null 2>&1; then
    echo "cliproxy=running pid=${cliproxy_pid} started_by_run=${cliproxy_started_by_run:-0}"
  else
    echo "cliproxy=stopped"
  fi

  if [[ -n "${tcpdump_pid:-}" ]] && kill -0 "${tcpdump_pid}" >/dev/null 2>&1; then
    echo "tcpdump=running pid=${tcpdump_pid}"
  else
    echo "tcpdump=stopped"
  fi

  docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | grep -E "(${snare_container}|${tanner_container}|${tanner_api_container}|${tanner_web_container}|${tanner_redis_container}|${tanner_phpox_container})" || true
}

list_runs() {
  if [[ ! -d "$CAPTURES_DIR" ]]; then
    echo "No capture directory found: ${CAPTURES_DIR}"
    return 0
  fi
  local run_info
  find "$CAPTURES_DIR" -maxdepth 2 -name run_info.env | sort | while read -r run_info; do
    echo "--- ${run_info%/run_info.env} ---"
    sed -n '1,8p' "$run_info"
  done
}

stop_legacy() {
  require_file "$BASE_TANNER_COMPOSE"
  require_file "$BASE_SNARE_COMPOSE"

  echo "[info] Stopping legacy base stack"
  run_compose -f "$BASE_SNARE_COMPOSE" down --remove-orphans
  run_compose -f "$BASE_TANNER_COMPOSE" down --remove-orphans

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] pkill -f 'tcpdump .*${ROOT_DIR}/captures/'"
    return 0
  fi

  if pgrep -af "tcpdump .*${ROOT_DIR}/captures/" >/dev/null 2>&1; then
    require_sudo_if_needed
    sudo pkill -f "tcpdump .*${ROOT_DIR}/captures/" || true
  fi

  echo "[ok] Legacy stack stop requested"
}

case "$COMMAND" in
  start)
    start_run
    ;;
  stop)
    stop_run
    ;;
  status)
    status_run
    ;;
  list)
    list_runs
    ;;
  stop-legacy)
    stop_legacy
    ;;
  -h|--help|"")
    usage
    ;;
  *)
    echo "Unknown command: ${COMMAND}" >&2
    usage
    exit 1
    ;;
esac
