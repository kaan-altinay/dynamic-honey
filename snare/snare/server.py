import asyncio
import logging
import aiohttp
import aiohttp_jinja2
import jinja2

from aiohttp import web
from aiohttp.web import StaticResource as StaticRoute
from urllib.parse import unquote
from snare.middlewares import SnareMiddleware
from snare.tanner_handler import TannerHandler


class HttpRequestHandler:
    def __init__(self, meta, run_args, snare_uuid, debug=False, keep_alive=75, **kwargs):
        self.run_args = run_args
        self.dir = run_args.full_page_path
        self.meta = meta
        self.snare_uuid = snare_uuid
        self.logger = logging.getLogger(__name__)
        self.sroute = StaticRoute(name=None, prefix="/", directory=self.dir)
        self.tanner_handler = TannerHandler(run_args, meta, snare_uuid)

    async def submit_slurp(self, data):
        try:
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(verify_ssl=False)) as session:
                r = await session.post(
                    "https://{0}:8080/api?auth={1}&chan=snare_test&msg={2}".format(
                        self.run_args.slurp_host, self.run_args.slurp_auth, data
                    ),
                    json=data,
                    timeout=10.0,
                )
                assert r.status == 200
                r.close()
        except Exception as e:
            self.logger.error("Error submitting slurp: %s", e)

    async def handle_request(self, request):
        self.logger.info("Request path: {0}".format(request.path_qs))
        data = self.tanner_handler.create_data(request, 200)
        
        # Meta Probe Logic
        base_path = request.path
        if not base_path.startswith("/"):
            base_path = "/" + base_path

        index_page = getattr(self.run_args, "index_page", "/index.html")
        if not index_page.startswith("/"):
            index_page = "/" + index_page

        candidates = []
        seen = set()

        def add(path: str):
            if not path:
                return

            normalized_path = path if path.startswith("/") else "/" + path
            if normalized_path not in seen:
                seen.add(normalized_path)
                candidates.append(normalized_path)

        # Account for root -> index page, and trailing slash + directory index variants.
        def add_path_variants(path: str):
            add(path)
            if path == "/":
                add(index_page)
                return

            if path.endswith("/"):
                trimmed_path = path[:-1] or "/"
                add(trimmed_path)
                add((trimmed_path if trimmed_path != "/" else "") + index_page)
                return

            add(path + "/")
            add(path + index_page)

        # Variants of base path.
        add_path_variants(base_path)

        # Variants of unquoted path.
        add_path_variants(unquote(base_path))

        matched_key = None
        hit = False
        if isinstance(self.meta, dict):
            for c in candidates:
                if c in self.meta:
                    matched_key = c
                    hit = True
                    break
            
        data["meta_probe"] = {
            "hit": hit,
            "matched_key": matched_key,
            "candidates": candidates,
            "index_page": index_page,
        }

        if request.method == "POST":
            post_data = await request.post()
            self.logger.info("POST data:")
            for key, val in post_data.items():
                self.logger.info("\t- {0}: {1}".format(key, val))
            data["post_data"] = dict(post_data)

        # Submit the event to the TANNER service
        event_result = await self.tanner_handler.submit_data(data)
        event_message = event_result.get("response", {}).get("message", {}) if event_result else {}
        meta_job_id = event_message.get("meta_job_id")
        if meta_job_id:
            asyncio.create_task(self.tanner_handler.poll_meta_job(meta_job_id, request.path_qs))

        # Log the event to slurp service if enabled
        if self.run_args.slurp_enabled:
            await self.submit_slurp(request.path_qs)

        detection = event_message.get("detection", {"name": "index", "order": 1, "type": 1})
        content, headers, status_code = await self.tanner_handler.parse_tanner_response(request.path_qs, detection)

        if self.run_args.server_header:
            headers["Server"] = self.run_args.server_header

        if "cookies" in data and "sess_uuid" in data["cookies"]:
            previous_sess_uuid = data["cookies"]["sess_uuid"]
        else:
            previous_sess_uuid = None

        if event_result is not None and "sess_uuid" in event_result["response"]["message"]:
            cur_sess_id = event_result["response"]["message"]["sess_uuid"]
            if previous_sess_uuid is None or not previous_sess_uuid.strip() or previous_sess_uuid != cur_sess_id:
                headers.add("Set-Cookie", "sess_uuid=" + cur_sess_id)

        return web.Response(body=content, status=status_code, headers=headers)

    async def start(self):
        app = web.Application()
        app.add_routes([web.route("*", "/{tail:.*}", self.handle_request)])
        aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader(self.dir))
        middleware = SnareMiddleware(
            error_404=self.meta["/status_404"].get("hash"),
            headers=self.meta["/status_404"].get("headers", []),
            server_header=self.run_args.server_header,
        )
        middleware.setup_middlewares(app)

        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.run_args.host_ip, self.run_args.port)

        await site.start()
        names = sorted(str(s.name) for s in self.runner.sites)
        print("======== Running on {} ========\n" "(Press CTRL+C to quit)".format(", ".join(names)))

    async def stop(self):
        await self.runner.cleanup()
