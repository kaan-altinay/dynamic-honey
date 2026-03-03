import asyncio
import base64
import json
import logging
import yarl
import uuid

from aiohttp import web

from tanner import dorks_manager, redis_client
from tanner.sessions import session_manager
from tanner.config import TannerConfig
from tanner.emulators import base
from tanner.generator import base_generator
from tanner.reporting.log_local import Reporting as local_report
from tanner.reporting.log_mongodb import Reporting as mongo_report
from tanner.reporting.log_hpfeeds import Reporting as hpfeeds_report
from tanner import __version__ as tanner_version

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
        self.generator = base_generator.BaseGenerator() 

        if TannerConfig.get("HPFEEDS", "enabled") is True:
            self.hpf = hpfeeds_report()
            self.hpf.connect()

            if self.hpf.connected() is False:
                self.logger.warning("hpfeeds not connected - no hpfeeds messages will be created")

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
            generation_result = await self.generator.generate_page(host=host, path=path, site_profile=site_profile)
            if not generation_result:
                raise NotImplementedError("Meta generation is not implemented for the current generator")

            body_bytes = generation_result.get("body_bytes")
            if not isinstance(body_bytes, (bytes, bytearray)):
                raise ValueError("Generator must return body_bytes as bytes")

            headers = generation_result.get("headers", [])
            page_path = generation_result.get("path", path)
            body_b64 = base64.b64encode(bytes(body_bytes)).decode("ascii")

            await self._save_meta_job(
                job_id,
                {
                    "state": "ready",
                    "path": page_path,
                    "headers": headers,
                    "body_b64": body_b64,
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
            await self.dorks.extract_path(path, self.redis_client)
            detection = await self.base_handler.handle(data, session)
            session.set_attack_type(path, detection["name"])

            meta_job_id = None
            meta_probe = data.get("meta_probe")
            meta_probe_hit = meta_probe.get("hit") if isinstance(meta_probe, dict) else None
            if meta_probe_hit is False:
                meta_job_id = str(uuid.uuid4())
                await self._save_meta_job(meta_job_id, {"state": "pending", "path": path})
                host = self._extract_host(data)
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

        headers = []
        headers_raw = job_data.get("headers")
        if headers_raw:
            try:
                headers = json.loads(headers_raw)
            except ValueError:
                self.logger.warning("Invalid headers payload for meta job %s", job_id)

        response_msg = self._make_response(
            msg={
                "state": "ready",
                "job_id": job_id,
                "path": job_data.get("path"),
                "headers": headers,
                "body_b64": job_data.get("body_b64", ""),
            }
        )
        return web.json_response(response_msg)

    async def on_shutdown(self, app):
        await self.session_manager.delete_sessions_on_shutdown(self.redis_client)
        await self.redis_client.close()

    async def delete_sessions(self):
        try:
            while True:
                await self.session_manager.delete_old_sessions(self.redis_client)
                await asyncio.sleep(self.delete_timeout)
        except asyncio.CancelledError:
            pass

    def setup_routes(self, app):
        app.router.add_route("*", "/", self.default_handler)
        app.router.add_post("/event", self.handle_event)
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
