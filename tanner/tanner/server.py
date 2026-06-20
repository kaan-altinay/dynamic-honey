import asyncio
import base64
import json
import logging
import re
import time
import yarl
import uuid

from collections import defaultdict, deque

from aiohttp import web

from tanner import dorks_manager, redis_client
from tanner.sessions import session_manager
from tanner.config import TannerConfig
from tanner.emulators import base
from tanner.generator import base_generator
from tanner.generator.agentic import AgenticBundleGenerator, GeneratedBundle
from tanner.generator.agentic.models import FlowDescriptor
from tanner.reporting.log_local import Reporting as local_report
from tanner.reporting.log_mongodb import Reporting as mongo_report
from tanner.reporting.log_hpfeeds import Reporting as hpfeeds_report
from tanner import __version__ as tanner_version
from tanner.flow import FlowEvaluator


class MetaGenerationPolicy:
    _HIGH_VALUE_PATH_RE = re.compile(
        r"(^/(api(?:/|$)|login|admin|wsman|_all_dbs|containers/json|solr(?:/|$)|v1/|_nodes|_stats|_cat/indices|json/version|_config|hnap1|tr064dev\.xml|nacos/|hazelcast/|clientwebservice/|manager/html|jmx|invoker/readonly|health|actuator/health))",
        re.IGNORECASE,
    )
    _STATIC_EXT_RE = re.compile(r"\.(?:png|jpg|jpeg|gif|svg|ico|css|js|map|woff2?|ttf|eot)(?:$|\?)", re.IGNORECASE)
    _EXPLOIT_PATH_RE = re.compile(
        r"\.\.|%2e|/bin/sh|php-cgi|eval-stdin|allow_url_include|auto_prepend_file|gponform|^/mcp$",
        re.IGNORECASE,
    )

    def __init__(self, logger):
        self.logger = logger
        self.enabled = self._as_bool(self._config_value("meta_policy_enabled", True))
        self.min_distinct_ips = int(self._config_value("meta_policy_min_distinct_ips", 2))
        self.min_recent_hits = int(self._config_value("meta_policy_min_recent_hits", 3))
        self.recent_window_seconds = int(self._config_value("meta_policy_recent_window_seconds", 86400))
        self.per_ip_cooldown_seconds = int(self._config_value("meta_policy_per_ip_cooldown_seconds", 1800))
        self.hourly_budget = int(self._config_value("meta_policy_hourly_budget", 10))
        self.daily_budget = int(self._config_value("meta_policy_daily_budget", 120))
        self.max_path_length = int(self._config_value("meta_policy_max_path_length", 120))
        pending_ttl_default = min(self.per_ip_cooldown_seconds, 600) if self.per_ip_cooldown_seconds > 0 else 600
        self.pending_ttl_seconds = int(self._config_value("meta_policy_pending_ttl_seconds", pending_ttl_default))

        self._path_events = defaultdict(deque)
        self._last_generation_by_ip = {}
        self._pending_paths = {}
        self._hourly_counts = defaultdict(int)
        self._daily_counts = defaultdict(int)

    @staticmethod
    def _as_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        if isinstance(value, (int, float)):
            return value != 0
        return bool(value)

    @staticmethod
    def _config_value(key, default):
        try:
            return TannerConfig.get("GENERATOR", key)
        except KeyError:
            return default

    @staticmethod
    def _looks_random_path(path):
        segments = [segment for segment in path.split("/") if segment]
        if not segments:
            return False
        token = max(segments, key=len)
        if token.startswith("."):
            return False
        if len(token) < 14:
            return False
        if not re.fullmatch(r"[A-Za-z0-9_-]+", token):
            return False
        unique_ratio = len(set(token)) / float(len(token))
        return unique_ratio > 0.65

    def _record_observation(self, path, src_ip, now):
        queue = self._path_events[path]
        queue.append((now, src_ip or ""))
        cutoff = now - self.recent_window_seconds
        while queue and queue[0][0] < cutoff:
            queue.popleft()

        recent_count = len(queue)
        distinct_ips = len({ip for _, ip in queue if ip})
        return recent_count, distinct_ips

    def _prune_pending_paths(self, now):
        if self.pending_ttl_seconds <= 0:
            return
        expired_paths = [path for path, ts in self._pending_paths.items() if now - ts > self.pending_ttl_seconds]
        for expired in expired_paths:
            self._pending_paths.pop(expired, None)

    def should_generate(self, path, method=None, src_ip=None, user_agent=None):
        if not self.enabled:
            return True, "policy_disabled"

        now = time.time()
        method = (method or "").upper()
        path = path if isinstance(path, str) and path else "/"
        src_ip = src_ip or ""
        user_agent = user_agent or ""

        recent_count, distinct_ips = self._record_observation(path, src_ip, now)
        self._prune_pending_paths(now)

        if method not in {"GET", "HEAD"}:
            return False, "method_not_allowed"
        if not path.startswith("/"):
            return False, "not_normalized_path"
        if len(path) < 1 or len(path) > self.max_path_length:
            return False, "path_length_filtered"
        if self._STATIC_EXT_RE.search(path):
            return False, "static_extension_filtered"
        if path in self._pending_paths:
            return False, "path_generation_pending"

        is_high_value = bool(self._HIGH_VALUE_PATH_RE.search(path))
        has_distinct_ips = distinct_ips >= self.min_distinct_ips
        has_recent_hits = recent_count >= self.min_recent_hits

        is_random = self._looks_random_path(path)
        exploit_like = bool(self._EXPLOIT_PATH_RE.search(path))

        if is_random or exploit_like:
            return False, "negative_signal"

        positive_signals = is_high_value or has_distinct_ips or has_recent_hits

        if not positive_signals:
            return False, "no_positive_signal"

        if src_ip and self.per_ip_cooldown_seconds > 0:
            last_generated = self._last_generation_by_ip.get(src_ip)
            if last_generated and (now - last_generated) < self.per_ip_cooldown_seconds:
                return False, "per_ip_cooldown"

        day_key = time.strftime("%Y-%m-%d", time.gmtime(now))
        hour_key = time.strftime("%Y-%m-%d %H", time.gmtime(now))

        if self.hourly_budget > 0 and self._hourly_counts[hour_key] >= self.hourly_budget:
            return False, "hourly_budget_exhausted"
        if self.daily_budget > 0 and self._daily_counts[day_key] >= self.daily_budget:
            return False, "daily_budget_exhausted"

        if src_ip:
            self._last_generation_by_ip[src_ip] = now
        self._hourly_counts[hour_key] += 1
        self._daily_counts[day_key] += 1
        self._pending_paths[path] = now
        return True, "scheduled"

class TannerServer:
    def __init__(self):
        base_dir = TannerConfig.get("EMULATORS", "root_dir")
        db_name = TannerConfig.get("SQLI", "db_name")

        self.session_manager = session_manager.SessionManager()
        self.delete_timeout = TannerConfig.get("SESSIONS", "delete_timeout")

        self.dorks = dorks_manager.DorksManager()
        self.base_handler = base.BaseHandler(base_dir, db_name)
        self.logger = logging.getLogger(__name__)
        self.redis_client = None
        self.generator = self._build_generator()
        self.flows_enabled = self._is_flows_enabled()
        self.flow_evaluator = FlowEvaluator() if self.flows_enabled else None
        self.meta_generation_policy = MetaGenerationPolicy(self.logger)
        self._registered_flow_descriptor_fingerprints = {}

        if TannerConfig.get("HPFEEDS", "enabled") is True:
            self.hpf = hpfeeds_report()
            self.hpf.connect()

            if self.hpf.connected() is False:
                self.logger.warning("hpfeeds not connected - no hpfeeds messages will be created")

    def _build_generator(self):
        try:
            backend = TannerConfig.get("GENERATOR", "backend")
        except KeyError:
            backend = None

        if isinstance(backend, str) and backend.strip().lower() == "agentic":
            self.logger.info("Using AgenticBundleGenerator backend")
            return AgenticBundleGenerator()
        return base_generator.BaseGenerator()

    def _is_flows_enabled(self):
        if not isinstance(self.generator, AgenticBundleGenerator):
            return False
        try:
            raw_value = TannerConfig.get("GENERATOR", "enable_scripted_flows")
        except KeyError:
            return False

        if isinstance(raw_value, bool):
            return raw_value
        if isinstance(raw_value, (int, float)):
            return raw_value != 0
        if isinstance(raw_value, str):
            return raw_value.strip().lower() in {"1", "true", "yes", "on"}
        return False

    @staticmethod
    def _make_response(msg):
        response_message = dict(version=tanner_version, response=dict(message=msg))
        return response_message

    @staticmethod
    async def default_handler(request):
        return web.Response(text="Tanner server")

    @staticmethod
    def _meta_job_key(job_id):
        return "meta_job:{}".format(job_id)

    @staticmethod
    def _extract_host(data):
        headers = data.get("headers") if isinstance(data, dict) else {}
        if isinstance(headers, dict):
            host = headers.get("Host")
            if isinstance(host, str) and host.strip():
                return host.split(":")[0]

        peer = data.get("peer") if isinstance(data, dict) else {}
        if isinstance(peer, dict):
            return peer.get("ip")
        return None

    @staticmethod
    def _serialize_generated_artifact(artifact):
        body_b64 = base64.b64encode(bytes(artifact.body_bytes)).decode("ascii")
        return {
            "path": artifact.path,
            "kind": artifact.kind,
            "headers": artifact.headers,
            "body_b64": body_b64,
            "status_code": artifact.status_code,
        }

    @staticmethod
    def _serialize_flow_descriptor(flow_descriptor):
        if flow_descriptor is None:
            return None
        if hasattr(flow_descriptor, "model_dump"):
            return flow_descriptor.model_dump(mode="json")
        if isinstance(flow_descriptor, dict):
            return flow_descriptor
        return None

    @staticmethod
    def _deserialize_json_field(raw_value, default):
        if raw_value is None or raw_value == "":
            return default
        if isinstance(raw_value, (dict, list)):
            return raw_value
        try:
            return json.loads(raw_value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _build_flow_detection(flow_result) -> dict:
        """Build a type-4 Tanner detection payload from a FlowMatchResult."""
        payload: dict = {"status_code": flow_result.status_code}
        if flow_result.redirect_to:
            payload["redirect_to"] = flow_result.redirect_to
        if flow_result.artifact_path:
            payload["rewritten_path"] = flow_result.artifact_path
        if flow_result.set_cookie:
            payload["set_cookie"] = flow_result.set_cookie
        if flow_result.clear_cookie:
            payload["clear_cookie"] = flow_result.clear_cookie
        if flow_result.headers:
            payload["headers"] = flow_result.headers
        return {"type": 4, "payload": payload}

    def _register_event_flow_descriptors(self, data):
        """Register persisted Snare flow descriptors carried with an event payload."""
        if not self.flows_enabled or self.flow_evaluator is None:
            return

        raw_descriptors = data.get("flow_descriptors") if isinstance(data, dict) else None
        if not isinstance(raw_descriptors, dict):
            return

        for key, raw_descriptor in raw_descriptors.items():
            if not isinstance(key, str) or not isinstance(raw_descriptor, dict):
                self.logger.warning("Ignoring malformed event flow descriptor entry for key %r", key)
                continue

            try:
                fingerprint = json.dumps(raw_descriptor, sort_keys=True, separators=(",", ":"))
                descriptor = FlowDescriptor.model_validate(raw_descriptor)
            except Exception as error:
                self.logger.warning("Ignoring invalid event flow descriptor for key %r: %s", key, error)
                continue

            if self._registered_flow_descriptor_fingerprints.get(key) == fingerprint:
                continue

            self.flow_evaluator.register(key, descriptor)
            self._registered_flow_descriptor_fingerprints[key] = fingerprint

    async def _save_meta_job(self, job_id, fields):
        if self.redis_client is None:
            return

        serialized_fields = {}
        for key, value in fields.items():
            if isinstance(value, (dict, list)):
                serialized_fields[key] = json.dumps(value)
            elif value is None:
                serialized_fields[key] = ""
            else:
                serialized_fields[key] = str(value)

        redis_key = self._meta_job_key(job_id)
        await self.redis_client.hset(redis_key, mapping=serialized_fields)
        await self.redis_client.expire(redis_key, 900)

    async def _run_meta_job(self, job_id, host, path, site_profile):
        try:
            generation_result = await self.generator.generate_bundle(host=host, path=path, site_profile=site_profile)
            if not generation_result:
                raise NotImplementedError("Meta generation is not implemented for the current generator")

            bundle = GeneratedBundle.model_validate(generation_result)
            allow_fallback = bool(
                getattr(getattr(self.generator, "runtime_config", None), "allow_fallback_persistence", False)
            )
            if bundle.used_fallback and not allow_fallback:
                raise ValueError("generated bundle used fallback; persistence is disabled")

            artifacts = [self._serialize_generated_artifact(artifact) for artifact in bundle.artifacts]
            flow_descriptor = self._serialize_flow_descriptor(getattr(bundle, "flow_descriptor", None))

            if self.flows_enabled and self.flow_evaluator is not None and flow_descriptor is not None:
                try:
                    self.flow_evaluator.register(bundle.primary_path, bundle.flow_descriptor)
                except Exception as flow_error:
                    self.logger.warning(
                        "Failed to register generated flow descriptor for %s: %s",
                        bundle.primary_path,
                        flow_error,
                    )
            await self._save_meta_job(
                job_id,
                {
                    "state": "ready",
                    "primary_path": bundle.primary_path,
                    "artifacts": artifacts,
                    "review_summary": bundle.review_summary,
                    "used_fallback": bundle.used_fallback,
                    "flow_descriptor": flow_descriptor,
                    "generation_trace": getattr(bundle, "generation_trace", []),
                    "generation_errors": getattr(bundle, "generation_errors", []),
                    "generation_diagnostics": getattr(bundle, "generation_diagnostics", []),
                },
            )
        except Exception as error:
            self.logger.exception("Meta job generation failed for %s", job_id)
            await self._save_meta_job(
                job_id,
                {
                    "state": "failed",
                    "path": path,
                    "error": str(error),
                },
            )

    async def handle_event(self, request):
        data = await request.read()
        try:
            data = json.loads(data.decode("utf-8"))
            path = yarl.URL(data["path"]).human_repr()
        except (TypeError, ValueError, KeyError) as error:
            self.logger.exception("error parsing request: %s", data)
            response_msg = self._make_response(msg=type(error).__name__)
        else:
            session, _ = await self.session_manager.add_or_update_session(data, self.redis_client)
            self.logger.info("Requested path %s", path)

            self._register_event_flow_descriptors(data)

            # V2: evaluate flow rules before normal detection
            if self.flows_enabled and self.flow_evaluator is not None:
                bare_path = path.split("?")[0] if "?" in path else path
                flow_result = self.flow_evaluator.evaluate(session, bare_path, data)
                if flow_result.matched:
                    detection = self._build_flow_detection(flow_result)
                    response_message = dict(detection=detection, sess_uuid=session.get_uuid())
                    response_msg = self._make_response(msg=response_message)
                    self.logger.info("Flow matched for %s -> type-4 detection", path)
                    session_data = data
                    session_data["response_msg"] = response_msg
                    if TannerConfig.get("LOCALLOG", "enabled") is True:
                        lr = local_report()
                        lr.create_session(session_data)
                    return web.json_response(response_msg)

            await self.dorks.extract_path(path, self.redis_client)
            detection = await self.base_handler.handle(data, session)
            session.set_attack_type(path, detection["name"])

            meta_job_id = None
            meta_probe = data.get("meta_probe")
            meta_probe_hit = meta_probe.get("hit") if isinstance(meta_probe, dict) else None
            if meta_probe_hit is False:
                host = self._extract_host(data)
                headers = data.get("headers") if isinstance(data.get("headers"), dict) else {}
                user_agent = headers.get("User-Agent", "")
                src_ip = data.get("peer", {}).get("ip") if isinstance(data.get("peer"), dict) else None

                should_generate, policy_reason = self.meta_generation_policy.should_generate(
                    path=path,
                    method=data.get("method"),
                    src_ip=src_ip,
                    user_agent=user_agent,
                )
                self.logger.info(
                    "Meta generation policy for path %s: allowed=%s reason=%s",
                    path,
                    should_generate,
                    policy_reason,
                )

                if should_generate:
                    meta_job_id = str(uuid.uuid4())
                    await self._save_meta_job(meta_job_id, {"state": "pending", "path": path})
                    asyncio.create_task(self._run_meta_job(meta_job_id, host, path, meta_probe))

                detection["type"] = 3
                detection["payload"] = {"status_code": 404}

            response_message = dict(detection=detection, sess_uuid=session.get_uuid())
            if meta_job_id is not None:
                response_message["meta_job_id"] = meta_job_id

            response_msg = self._make_response(msg=response_message)
            self.logger.info("TANNER response %s", response_msg)

            session_data = data
            session_data["response_msg"] = response_msg

            # Log to Mongo
            if TannerConfig.get("MONGO", "enabled") is True:
                db = mongo_report()
                session_id = db.create_session(session_data)
                self.logger.info("Writing session to DB: {}".format(session_id))

            # Log to hpfeeds
            if TannerConfig.get("HPFEEDS", "enabled") is True:
                if self.hpf.connected():
                    self.hpf.create_session(session_data)

            if TannerConfig.get("LOCALLOG", "enabled") is True:
                lr = local_report()
                lr.create_session(session_data)

        return web.json_response(response_msg)

    async def handle_meta_generate(self, request):
        data = await request.read()
        try:
            data = json.loads(data.decode("utf-8"))
            path = yarl.URL(data["path"]).human_repr()
        except (TypeError, ValueError, KeyError) as error:
            self.logger.exception("error parsing meta generate request: %s", data)
            response_msg = self._make_response(msg=type(error).__name__)
            return web.json_response(response_msg, status=400)

        host = data.get("host")
        if not isinstance(host, str) or not host.strip():
            host = None

        site_profile = data.get("site_profile")
        if not isinstance(site_profile, dict):
            site_profile = {}

        index_page = data.get("index_page")
        if isinstance(index_page, str) and index_page.strip():
            site_profile["index_page"] = index_page
        else:
            site_profile.setdefault("index_page", "/index.html")

        meta_job_id = str(uuid.uuid4())
        await self._save_meta_job(meta_job_id, {"state": "pending", "path": path})
        asyncio.create_task(self._run_meta_job(meta_job_id, host, path, site_profile))

        response_msg = self._make_response(
            msg={"state": "pending", "meta_job_id": meta_job_id, "path": path}
        )
        return web.json_response(response_msg, status=202)

    async def handle_dorks(self, request):
        dorks = await self.dorks.choose_dorks(self.redis_client)
        response_msg = dict(version=tanner_version, response=dict(dorks=dorks))
        return web.json_response(response_msg)

    async def handle_version(self, request):
        response_msg = dict(version=tanner_version)
        return web.json_response(response_msg)

    async def handle_meta_job(self, request):
        job_id = request.match_info["job_id"]
        if self.redis_client is None:
            response_msg = self._make_response(msg={"state": "failed", "error": "Redis client is not initialized"})
            return web.json_response(response_msg, status=500)

        job_data = await self.redis_client.hgetall(self._meta_job_key(job_id))
        if not job_data:
            response_msg = self._make_response(msg={"state": "missing", "job_id": job_id})
            return web.json_response(response_msg, status=404)

        state = job_data.get("state", "pending")
        if state == "pending":
            response_msg = self._make_response(msg={"state": "pending", "job_id": job_id})
            return web.json_response(response_msg, status=202)

        if state == "failed":
            response_msg = self._make_response(
                msg={
                    "state": "failed",
                    "job_id": job_id,
                    "error": job_data.get("error", "Unknown generation error"),
                }
            )
            return web.json_response(response_msg, status=500)

        artifacts = self._deserialize_json_field(job_data.get("artifacts"), [])
        flow_descriptor = self._deserialize_json_field(job_data.get("flow_descriptor"), None)
        generation_trace = self._deserialize_json_field(job_data.get("generation_trace"), [])
        generation_errors = self._deserialize_json_field(job_data.get("generation_errors"), [])
        generation_diagnostics = self._deserialize_json_field(job_data.get("generation_diagnostics"), [])
        response_msg = self._make_response(
            msg={
                "state": "ready",
                "job_id": job_id,
                "primary_path": job_data.get("primary_path"),
                "artifacts": artifacts,
                "review_summary": job_data.get("review_summary", ""),
                "used_fallback": str(job_data.get("used_fallback", "")).lower() == "true",
                "flow_descriptor": flow_descriptor,
                "generation_trace": generation_trace,
                "generation_errors": generation_errors,
                "generation_diagnostics": generation_diagnostics,
            }
        )
        return web.json_response(response_msg)

    async def on_shutdown(self, app):
        await self.session_manager.delete_sessions_on_shutdown(self.redis_client)
        await self.redis_client.close()

    async def delete_sessions(self):
        try:
            delay = self.delete_timeout if isinstance(self.delete_timeout, (int, float)) and self.delete_timeout > 0 else 300
            while True:
                await self.session_manager.delete_old_sessions(self.redis_client)
                await asyncio.sleep(delay)
        except asyncio.CancelledError:
            pass

    def setup_routes(self, app):
        app.router.add_route("*", "/", self.default_handler)
        app.router.add_post("/event", self.handle_event)
        app.router.add_post("/meta_generate", self.handle_meta_generate)
        app.router.add_get("/dorks", self.handle_dorks)
        app.router.add_get("/meta_job/{job_id}", self.handle_meta_job)
        app.router.add_get("/version", self.handle_version)

    async def make_app(self):
        app = web.Application()
        app.on_shutdown.append(self.on_shutdown)
        self.setup_routes(app)
        app.on_startup.append(self.start_background_delete)
        app.on_cleanup.append(self.cleanup_background_tasks)
        return app

    async def start_background_delete(self, app):
        app["session_delete"] = asyncio.ensure_future(self.delete_sessions())

    async def cleanup_background_tasks(self, app):
        app["session_delete"].cancel()
        await app["session_delete"]

    def start(self):
        loop = asyncio.get_event_loop()
        self.redis_client = loop.run_until_complete(redis_client.RedisClient.get_redis_client())

        host = TannerConfig.get("TANNER", "host")
        port = TannerConfig.get("TANNER", "port")

        web.run_app(self.make_app(), host=host, port=port)
