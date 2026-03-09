import asyncio
import base64
import hashlib
import re
import os
import multidict
import json
import logging
import aiohttp

from urllib.parse import unquote
from bs4 import BeautifulSoup
from snare.html_handler import HtmlHandler


class TannerHandler:
    def __init__(self, run_args, meta, snare_uuid):
        self.run_args = run_args
        self.meta = meta
        self.dir = run_args.full_page_path
        self.snare_uuid = snare_uuid
        self.html_handler = HtmlHandler(run_args.no_dorks, run_args.tanner)
        self.logger = logging.getLogger(__name__)
        self._meta_lock = asyncio.Lock()

    def create_data(self, request, response_status):
        data = dict(
            method=None,
            path=None,
            headers=None,
            uuid=self.snare_uuid.decode("utf-8"),
            peer=None,
            status=response_status,
        )
        if request.transport:
            peer = dict(
                ip=request.transport.get_extra_info("peername")[0],
                port=request.transport.get_extra_info("peername")[1],
            )
            data["peer"] = peer
        if request.path:
            # FIXME request.headers is a CIMultiDict, so items with the same
            # key will be overwritten when converting to dictionary
            header = {key: value for (key, value) in request.headers.items()}
            data["method"] = request.method
            data["headers"] = header
            data["path"] = request.path_qs
            if "Cookie" in header:
                data["cookies"] = {cookie.split("=")[0]: cookie.split("=")[1] for cookie in header["Cookie"].split(";")}
        return data

    async def submit_data(self, data):
        event_result = None
        try:
            async with aiohttp.ClientSession() as session:
                r = await session.post(
                    "http://{0}:8090/event".format(self.run_args.tanner),
                    json=data,
                    timeout=10.0,
                )
                try:
                    event_result = await r.json()
                except (
                    json.decoder.JSONDecodeError,
                    aiohttp.client_exceptions.ContentTypeError,
                ) as e:
                    self.logger.error("Error submitting data: {} {}".format(e, data))
                    event_result = {
                        "version": "0.6.0",
                        "response": {
                            "message": {
                                "detection": {
                                    "name": "index",
                                    "order": 1,
                                    "type": 1,
                                    "version": "0.6.0",
                                },
                                "sess_uuid": data["uuid"],
                            }
                        },
                    }
                finally:
                    await r.release()
        except Exception as e:
            self.logger.exception("Exception: %s", e)
            raise e
        return event_result

    @staticmethod
    def _normalize_headers(headers):
        if isinstance(headers, dict):
            return [{key: value} for key, value in headers.items()]
        if isinstance(headers, list):
            normalized_headers = []
            for header in headers:
                if isinstance(header, dict):
                    normalized_headers.append(header)
            return normalized_headers
        return []

    def _normalize_meta_path(self, requested_path):
        normalized_path = requested_path.split("?", 1)[0]
        normalized_path = unquote(normalized_path)
        if not normalized_path.startswith("/"):
            normalized_path = "/" + normalized_path
        if normalized_path == "/":
            return getattr(self.run_args, "index_page", "/index.html")
        if normalized_path.endswith("/"):
            return normalized_path[:-1]
        return normalized_path

    def parse_seed_endpoints(self, seed_endpoints_path):
        if not seed_endpoints_path:
            return []

        parsed_endpoints = []
        seen = set()
        with open(seed_endpoints_path) as seed_fh:
            for endpoint in seed_fh:
                endpoint = endpoint.strip()
                if not endpoint or endpoint.startswith("#"):
                    continue

                normalized_endpoint = self._normalize_meta_path(endpoint)
                if normalized_endpoint in seen:
                    continue
                seen.add(normalized_endpoint)
                parsed_endpoints.append(normalized_endpoint)
        return parsed_endpoints

    async def _request_meta_generate_job(self, requested_path):
        host = getattr(self.run_args, "host_ip", None)
        if not isinstance(host, str) or not host.strip():
            host = None

        payload = {
            "path": requested_path,
            "index_page": getattr(self.run_args, "index_page", "/index.html"),
            "site_profile": {
                "index_page": getattr(self.run_args, "index_page", "/index.html"),
                "candidates": [requested_path],
            },
        }
        if host is not None:
            payload["host"] = host

        endpoint = "http://{0}:8090/meta_generate".format(self.run_args.tanner)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, json=payload, timeout=10.0) as response:
                    try:
                        response_data = await response.json()
                    except (
                        json.decoder.JSONDecodeError,
                        aiohttp.client_exceptions.ContentTypeError,
                    ) as error:
                        self.logger.warning(
                            "Seed endpoint %s meta_generate response decode failed: %s",
                            requested_path,
                            error,
                        )
                        return None

                    if response.status >= 400:
                        self.logger.warning(
                            "Seed endpoint %s meta_generate returned status %s: %s",
                            requested_path,
                            response.status,
                            response_data,
                        )
                        return None
        except (aiohttp.ClientError, asyncio.TimeoutError) as error:
            self.logger.warning("Seed endpoint %s meta_generate request failed: %s", requested_path, error)
            return None

        message = response_data.get("response", {}).get("message", {}) if isinstance(response_data, dict) else {}
        return message.get("meta_job_id")

    async def consume_seed_endpoints(self, seed_endpoints):
        summary = {"requested": 0, "skipped": 0, "generated": 0, "failed": 0}
        if not seed_endpoints:
            return summary

        for endpoint in seed_endpoints:
            requested_path = self._normalize_meta_path(endpoint)
            summary["requested"] += 1

            if requested_path in self.meta:
                summary["skipped"] += 1
                continue

            meta_job_id = await self._request_meta_generate_job(requested_path)
            if not meta_job_id:
                self.logger.warning("Seed endpoint %s did not return meta job id", requested_path)
                summary["failed"] += 1
                continue

            try:
                is_generated = await self.poll_meta_job(meta_job_id, requested_path)
            except Exception as error:
                self.logger.warning("Seed endpoint %s generation failed: %s", requested_path, error)
                summary["failed"] += 1
                continue

            if is_generated:
                summary["generated"] += 1
            else:
                summary["failed"] += 1

        return summary
    async def _save_generated_meta(self, requested_path, headers, body_bytes):
        content_hash = hashlib.md5(body_bytes).hexdigest()
        content_file = os.path.join(self.dir, content_hash)
        with open(content_file, "wb") as generated_content:
            generated_content.write(body_bytes)

        meta_key = self._normalize_meta_path(requested_path)
        normalized_headers = self._normalize_headers(headers)

        async with self._meta_lock:
            self.meta[meta_key] = {"hash": content_hash, "headers": normalized_headers}
            meta_path = os.path.join(self.dir, "meta.json")
            temp_meta_path = "{}.tmp".format(meta_path)
            with open(temp_meta_path, "w") as meta_file:
                json.dump(self.meta, meta_file)
            os.replace(temp_meta_path, meta_path)

    async def poll_meta_job(self, meta_job_id, requested_path, poll_interval=1.0, max_attempts=30):
        endpoint = "http://{0}:8090/meta_job/{1}".format(self.run_args.tanner, meta_job_id)
        async with aiohttp.ClientSession() as session:
            for _ in range(max_attempts):
                try:
                    async with session.get(endpoint, timeout=10.0) as response:
                        if response.status == 404:
                            self.logger.warning("Meta job %s not found", meta_job_id)
                            return False

                        response_data = await response.json()
                except (
                    json.decoder.JSONDecodeError,
                    aiohttp.client_exceptions.ContentTypeError,
                    aiohttp.ClientError,
                    asyncio.TimeoutError,
                ) as error:
                    self.logger.warning("Polling meta job %s failed: %s", meta_job_id, error)
                    await asyncio.sleep(poll_interval)
                    continue

                message = response_data.get("response", {}).get("message", {})
                state = message.get("state")
                if response.status == 202 or state == "pending":
                    await asyncio.sleep(poll_interval)
                    continue

                if state == "ready":
                    body_b64 = message.get("body_b64")
                    if not body_b64:
                        self.logger.warning("Meta job %s returned ready state without body", meta_job_id)
                        return False

                    try:
                        body_bytes = base64.b64decode(body_b64)
                    except ValueError as error:
                        self.logger.warning("Meta job %s body decode failed: %s", meta_job_id, error)
                        return False

                    await self._save_generated_meta(
                        requested_path=message.get("path", requested_path),
                        headers=message.get("headers", []),
                        body_bytes=body_bytes,
                    )
                    self.logger.info("Stored generated meta content for path %s", requested_path)
                    return True

                if state == "failed":
                    self.logger.warning("Meta job %s failed: %s", meta_job_id, message.get("error"))
                    return False

                self.logger.warning("Meta job %s returned unexpected state payload: %s", meta_job_id, message)
                return False

        self.logger.warning("Meta job %s timed out after %s attempts", meta_job_id, max_attempts)
        return False

    async def parse_tanner_response(self, requested_name, detection):
        content = None
        status_code = 200
        headers = multidict.CIMultiDict()
        # Creating a regex object for the pattern of multiple contiguous forward slashes
        p = re.compile("/+")
        # Substituting all occurrences of the pattern with single forward slash
        requested_name = p.sub("/", requested_name)

        if detection["type"] == 1:
            possible_requests = [requested_name]
            query_start = requested_name.find("?")
            if query_start != -1:
                possible_requests.append(requested_name[:query_start])

            file_name = None
            for requested_name in possible_requests:
                if requested_name == "/":
                    requested_name = self.run_args.index_page
                if requested_name[-1] == "/":
                    requested_name = requested_name[:-1]
                requested_name = unquote(requested_name)
                try:
                    file_name = self.meta[requested_name]["hash"]
                    for header in self.meta[requested_name].get("headers", []):
                        for key, value in header.items():
                            headers.add(key, value)
                    # overwrite headers with legacy content-type if present and not none
                    content_type = self.meta[requested_name].get("content_type")
                    if content_type:
                        headers["Content-Type"] = content_type
                except KeyError:
                    pass
                else:
                    break

            if not file_name:
                status_code = 404
            else:
                path = os.path.join(self.dir, file_name)
                if os.path.isfile(path):
                    with open(path, "rb") as fh:
                        content = fh.read()
                    if headers.get("Content-Type", "").startswith("text/html"):
                        content = await self.html_handler.handle_content(content)

        elif detection["type"] == 2:
            payload_content = detection["payload"]
            if payload_content["page"]:
                try:
                    file_name = self.meta[payload_content["page"]]["hash"]
                    for header in self.meta[payload_content["page"]].get("headers", []):
                        for key, value in header.items():
                            headers.add(key, value)
                    # overwrite headers with legacy content-type if present and not none
                    content_type = self.meta[payload_content["page"]].get("content_type")
                    if content_type:
                        headers["Content-Type"] = content_type
                    page_path = os.path.join(self.dir, file_name)
                    with open(page_path, encoding="utf-8") as p:
                        content = p.read()
                except KeyError:
                    content = "<html><body></body></html>"
                    headers["Content-Type"] = "text/html"

                soup = BeautifulSoup(content, "html.parser")
                script_tag = soup.new_tag("div")
                script_tag.append(BeautifulSoup(payload_content["value"], "html.parser"))
                soup.body.append(script_tag)
                content = str(soup).encode()
            else:
                content_type = "text/plain"
                if content_type:
                    headers["Content-Type"] = content_type
                content = payload_content["value"].encode("utf-8")

            if "headers" in payload_content:
                # overwrite local headers with the tanner-provided ones
                headers.update(payload_content["headers"])

        else:  # type 3
            payload_content = detection["payload"]
            status_code = payload_content["status_code"]

        return content, headers, status_code
