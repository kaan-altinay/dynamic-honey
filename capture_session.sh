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
  --mode <name>           default | agentic | current | cache (default: agentic)
                          default  -> same source tree, but with GENERATOR.backend forced to none via temp config.
                          agentic/current/cache -> use the repo config as-is.
  --range <start-end>     Public IP host suffix range, must be a contiguous block of 4 aligned on 16,20,24,28.
                          Example: 16-19
  --range-start <start>   Equivalent shorthand for a 4-IP block. Example: --range-start 16 means 16-19.
  --captures-dir <path>   Artifact root (default: ./captures_new)
  --iface <name>          Capture interface (default: eth0)
  --page-url <host>       Snare page-dir / PAGE_URL (default: example.com)
  --build                 Rebuild images before starting services
  --dry-run               Print commands without executing
  -h, --help              Show this help

Notes:
  - Start one instance per 4-IP range. Up to four concurrent runs can cover 16-19, 20-23, 24-27, and 28-31.
  - This script uses the same source tree for every run. Separate run directories keep captures, runtime files,
    and snare state isolated.
  - The "default" mode disables the agentic backend via a temporary Tanner config overlay. It does not restore a
    pristine upstream source tree; for that you would need a separate clean checkout or image.
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
DEFAULT_IFACE="eth0"

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
    *)
      echo "Unsupported mode: $1" >&2
      exit 1
      ;;
  esac
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

prepare_run_dirs() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi
  mkdir -p "$RUN_DIR" "$RUNTIME_DIR" "$STATE_DIR"
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
      - /data

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
    command: ["/opt/tanner/tanner-env/bin/tanner", "--config", "/opt/tanner/runtime-config/config.yaml"]
    volumes:
      - '${TANNER_CONFIG_PATH}:/opt/tanner/runtime-config/config.yaml:ro'
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
  local ip
  for ip in "${IP_LIST[@]}"; do
    ports_block+="      - '${ip}:80:80'"$'\n'
  done

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] write Snare compose -> ${SNARE_COMPOSE_PATH}"
    return 0
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
      - TANNER=tanner
      - PAGE_URL=${PAGE_URL}
      - PORT=80
    volumes:
      - '${STATE_DIR}:/opt/snare'

networks:
  local:
    external: true
    name: ${NETWORK_NAME}
EOF
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
      run_name|mode|project_name|network_name|range_start|range_end|ip_list|iface|page_url|web_port|tanner_compose|snare_compose|tanner_config|state_dir|pcap_path|tcpdump_log_path|snare_container|tanner_container|tanner_api_container|tanner_web_container|tanner_redis_container|tanner_phpox_container|tcpdump_pid)
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
  build_tcpdump_filter
  init_run_layout

  if [[ -e "$RUN_DIR" ]]; then
    echo "Run directory already exists: $RUN_DIR" >&2
    echo "Use a new --run-name or stop/remove the existing run first." >&2
    exit 1
  fi

  prepare_run_dirs
  seed_snare_state
  write_tanner_config "$MODE"
  write_tanner_compose
  write_snare_compose
  write_run_info

  echo "[info] Starting run '${RUN_NAME}' (${MODE}) for ${PUBLIC_IP_PREFIX}.${RANGE_START}-${RANGE_END}"
  echo "[info] Capture dir: ${RUN_DIR}"
  echo "[info] Snare state: ${STATE_DIR}"
  echo "[info] Tanner web: http://127.0.0.1:${WEB_PORT}"
  echo "[info] tcpdump filter: ${TCPDUMP_FILTER}"

  run_compose -p "$PROJECT_NAME" -f "$TANNER_COMPOSE_PATH" down --remove-orphans
  run_compose -p "$PROJECT_NAME" -f "$SNARE_COMPOSE_PATH" down --remove-orphans

  if [[ "$BUILD" -eq 1 ]]; then
    run_compose -p "$PROJECT_NAME" -f "$TANNER_COMPOSE_PATH" up -d --build
    run_compose -p "$PROJECT_NAME" -f "$SNARE_COMPOSE_PATH" up -d --build
  else
    run_compose -p "$PROJECT_NAME" -f "$TANNER_COMPOSE_PATH" up -d
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
  write_container_logs "$tanner_container" "${RUN_DIR}/tanner.docker.log"
  write_container_logs "$tanner_api_container" "${RUN_DIR}/tanner_api.docker.log"
  write_container_logs "$tanner_web_container" "${RUN_DIR}/tanner_web.docker.log"
  write_container_logs "$tanner_redis_container" "${RUN_DIR}/tanner_redis.docker.log"
  write_container_logs "$tanner_phpox_container" "${RUN_DIR}/tanner_phpox.docker.log"

  run_compose -p "$project_name" -f "$snare_compose" down --remove-orphans
  run_compose -p "$project_name" -f "$tanner_compose" down --remove-orphans

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
