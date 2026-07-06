import asyncio
import datetime
import time
import base64
import binascii
import hashlib
import re
import os
import multidict
import json
import logging
import aiohttp

from urllib.parse import unquote, urlsplit
from bs4 import BeautifulSoup
from snare.html_handler import HtmlHandler


META_JOB_POLL_INTERVAL_SECONDS = 3.0
META_JOB_MAX_ATTEMPTS = 600

# Headers that exist purely for internal bookkeeping (e.g. tagging which
# generator produced an artifact) and must never reach a real HTTP response
# -- they'd let an attacker trivially fingerprint generated content. Old
# cached meta.json entries may already carry these; filtering here (the
# single choke point all served responses pass through) covers both
# already-persisted caches and anything generated going forward, with no
# need to rewrite any cache files.
_INTERNAL_ONLY_HEADERS = ("X-Tanner-Generated",)

# Internal flow-bookkeeping path prefix. Artifacts under /_flow/ are
# server-internal routing keys the V2 flow evaluator uses to look up which
# variant to serve next (e.g. /_flow/boaform-admin-formLogin/post-invalid)
# -- they were never meant to be literal, attacker-visible URLs. Some
# already-persisted bundles nonetheless link to them directly from rendered
# HTML/JS (nav links, redirect scripts), which is itself a honeypot tell.
# Stripping the prefix at the same single choke point as the header
# filtering above sanitizes both already-cached content and anything
# generated going forward, with no need to rewrite any cache files.
_INTERNAL_FLOW_PATH_PREFIX = "/_flow/"


class TannerHandler:
    def __init__(self, run_args, meta, snare_uuid):
        self.run_args = run_args
        self.meta = meta
        self.dir = run_args.full_page_path
        self.snare_uuid = snare_uuid
        self.html_handler = HtmlHandler(run_args.no_dorks, run_args.tanner)
        self.logger = logging.getLogger(__name__)
        self._meta_lock = asyncio.Lock()
        self.flow_descriptors = self._load_persisted_flow_descriptors()

    def _load_persisted_flow_descriptors(self):
        flows_path = os.path.join(self.dir, "flows.json")
        try:
            return self._load_flow_descriptors(flows_path)
        except (OSError, ValueError) as error:
            self.logger.warning("Failed to load persisted flow descriptors from %s: %s", flows_path, error)
            return {}

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
        if self.flow_descriptors:
            data["flow_descriptors"] = self.flow_descriptors
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

    @staticmethod
    def _is_external_seed_url(seed_value):
        parsed_seed = urlsplit(seed_value)
        return bool(parsed_seed.scheme and parsed_seed.netloc)

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

                if self._is_external_seed_url(endpoint):
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
    @staticmethod
    def _decode_generated_body(body_b64):
        if not isinstance(body_b64, str) or not body_b64:
            raise ValueError("Generated artifact is missing body_b64")
        try:
            return base64.b64decode(body_b64, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError("Generated artifact body decode failed: {}".format(error))

    def _prepare_generated_artifact(self, artifact, default_requested_path):
        if not isinstance(artifact, dict):
            raise ValueError("Generated artifact payload must be an object")

        status_code = artifact.get("status_code", 200)
        if status_code != 200:
            raise ValueError("Generated artifact status code must be 200")

        normalized_path = self._normalize_meta_path(artifact.get("path", default_requested_path))
        body_bytes = self._decode_generated_body(artifact.get("body_b64"))
        normalized_headers = self._normalize_headers(artifact.get("headers", []))
        content_hash = hashlib.md5(body_bytes).hexdigest()

        return {
            "path": normalized_path,
            "headers": normalized_headers,
            "body_bytes": body_bytes,
            "hash": content_hash,
        }

    def _load_flow_descriptors(self, flows_path):
        if not os.path.exists(flows_path):
            return {}
        with open(flows_path) as flows_file:
            flow_descriptors = json.load(flows_file)
        if not isinstance(flow_descriptors, dict):
            raise ValueError("flows.json must contain an object")
        return flow_descriptors

    def _prepare_flow_descriptor(self, requested_path, flow_descriptor):
        if flow_descriptor is None:
            return None
        if not isinstance(flow_descriptor, dict):
            raise ValueError("Flow descriptor payload must be an object")
        if "rules" not in flow_descriptor or not isinstance(flow_descriptor.get("rules"), list):
            raise ValueError("Flow descriptor payload must contain rules list")
        return {
            "path": self._normalize_meta_path(requested_path),
            "descriptor": flow_descriptor,
        }

    def _write_prepared_flow_descriptor(self, prepared_flow_descriptor):
        if prepared_flow_descriptor is None:
            return
        flows_path = os.path.join(self.dir, "flows.json")
        updated_flows = self._load_flow_descriptors(flows_path)
        updated_flows[prepared_flow_descriptor["path"]] = prepared_flow_descriptor["descriptor"]
        temp_flows_path = "{}.tmp".format(flows_path)
        with open(temp_flows_path, "w") as flows_file:
            json.dump(updated_flows, flows_file, indent=2, sort_keys=True)
        os.replace(temp_flows_path, flows_path)
        self.flow_descriptors = updated_flows

    def _write_generation_report(self, message):
        """Write a per-endpoint generation report JSON alongside meta.json for debugging."""
        primary_path = message.get("primary_path") or ""
        # Build a slug from the primary path for use as filename.
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", primary_path.strip("/")) or "root"
        slug = slug[:120]

        reports_dir = os.path.join(self.dir, "generation_reports")
        try:
            os.makedirs(reports_dir, exist_ok=True)
        except OSError as error:
            self.logger.warning("Failed to create generation_reports dir: %s", error)
            return

        artifacts = message.get("artifacts") or []
        report = {
            "timestamp": datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat(),
            "primary_path": primary_path,
            "used_fallback": message.get("used_fallback", False),
            "review_summary": message.get("review_summary", ""),
            "artifact_count": len(artifacts),
            "artifacts": [
                {
                    "path": a.get("path"),
                    "kind": a.get("kind"),
                    "bytes": len(base64.b64decode(a["body_b64"])) if isinstance(a.get("body_b64"), str) else None,
                }
                for a in artifacts
                if isinstance(a, dict)
            ],
            "generation_trace": message.get("generation_trace") or [],
            "generation_errors": message.get("generation_errors") or [],
            "generation_diagnostics": message.get("generation_diagnostics") or [],
        }

        report_path = os.path.join(reports_dir, "{}.json".format(slug))
        tmp_path = "{}.tmp".format(report_path)
        try:
            with open(tmp_path, "w") as fh:
                json.dump(report, fh, indent=2, sort_keys=True)
            os.replace(tmp_path, report_path)
        except OSError as error:
            self.logger.warning("Failed to write generation report for %s: %s", primary_path, error)


    async def _save_generated_artifacts(self, artifacts, requested_path, flow_descriptor=None):
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError("Meta job ready payload did not include artifacts")

        prepared_artifacts = []
        seen_paths = set()
        for artifact in artifacts:
            prepared_artifact = self._prepare_generated_artifact(artifact, requested_path)
            if prepared_artifact["path"] in seen_paths:
                raise ValueError("Meta job bundle contains duplicate path {}".format(prepared_artifact["path"]))
            seen_paths.add(prepared_artifact["path"])
            prepared_artifacts.append(prepared_artifact)
        prepared_flow_descriptor = self._prepare_flow_descriptor(requested_path, flow_descriptor)

        for prepared_artifact in prepared_artifacts:
            content_file = os.path.join(self.dir, prepared_artifact["hash"])
            # Hash-addressed content is immutable; if already present, reuse it.
            if os.path.exists(content_file):
                continue
            with open(content_file, "wb") as generated_content:
                generated_content.write(prepared_artifact["body_bytes"])

        async with self._meta_lock:
            updated_meta = dict(self.meta)
            for prepared_artifact in prepared_artifacts:
                updated_meta[prepared_artifact["path"]] = {
                    "hash": prepared_artifact["hash"],
                    "headers": prepared_artifact["headers"],
                }

            meta_path = os.path.join(self.dir, "meta.json")
            temp_meta_path = "{}.tmp".format(meta_path)
            with open(temp_meta_path, "w") as meta_file:
                json.dump(updated_meta, meta_file)
            os.replace(temp_meta_path, meta_path)
            self.meta.clear()
            self.meta.update(updated_meta)
            self._write_prepared_flow_descriptor(prepared_flow_descriptor)

    async def poll_meta_job(
        self,
        meta_job_id,
        requested_path,
        poll_interval=META_JOB_POLL_INTERVAL_SECONDS,
        max_attempts=META_JOB_MAX_ATTEMPTS,
    ):
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
                    artifacts = message.get("artifacts")
                    try:
                        await self._save_generated_artifacts(
                            artifacts=artifacts,
                            requested_path=message.get("primary_path", requested_path),
                            flow_descriptor=message.get("flow_descriptor"),
                        )
                    except ValueError as error:
                        self.logger.warning("Meta job %s returned invalid bundle: %s", meta_job_id, error)
                        return False
                    except OSError as error:
                        self.logger.warning("Meta job %s could not be persisted: %s", meta_job_id, error)
                        return False

                    self._write_generation_report(message)
                    self.logger.info(
                        "Stored generated meta bundle for path %s with %s artifacts",
                        requested_path,
                        len(artifacts),
                    )
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

        elif detection["type"] == 3:
            payload_content = detection["payload"]
            status_code = payload_content["status_code"]
        elif detection["type"] == 4:
            # V2 scripted-flow response
            payload_content = detection["payload"]
            status_code = payload_content.get("status_code", 200)

            # Extra headers
            for key, value in payload_content.get("headers", {}).items():
                headers.add(key, value)

            # Set cookies
            for name, val in payload_content.get("set_cookie", {}).items():
                headers.add("Set-Cookie", "{}={}; Path=/; HttpOnly".format(name, val))

            # Clear cookies
            for name in payload_content.get("clear_cookie", []):
                headers.add("Set-Cookie", "{}=; Path=/; Max-Age=0; HttpOnly".format(name))

            redirect_to = payload_content.get("redirect_to")
            rewritten_path = payload_content.get("rewritten_path")

            if redirect_to:
                headers.add("Location", redirect_to)
            elif rewritten_path:
                file_name = None
                try:
                    file_name = self.meta[rewritten_path]["hash"]
                    for header in self.meta[rewritten_path].get("headers", []):
                        for key, value in header.items():
                            headers.add(key, value)
                    content_type = self.meta[rewritten_path].get("content_type")
                    if content_type:
                        headers["Content-Type"] = content_type
                except KeyError:
                    pass
                if file_name:
                    path = os.path.join(self.dir, file_name)
                    if os.path.isfile(path):
                        with open(path, "rb") as fh:
                            content = fh.read()
                        if headers.get("Content-Type", "").startswith("text/html"):
                            content = await self.html_handler.handle_content(content)
                    else:
                        status_code = 404
                else:
                    status_code = 404

        for header_name in _INTERNAL_ONLY_HEADERS:
            headers.popall(header_name, None)
        content = self._strip_internal_flow_references(content)
        return content, headers, status_code

    @staticmethod
    def _strip_internal_flow_references(content):
        """Remove the /_flow/ internal-routing prefix from served content.

        Applied unconditionally to every response leaving parse_tanner_response
        (cache hits and freshly generated content alike) so a leaked literal
        reference -- e.g. an <a href="/_flow/foo/post-invalid"> nav link or an
        inline redirect script -- never reaches the client as-is.
        """
        if isinstance(content, bytes):
            return content.replace(_INTERNAL_FLOW_PATH_PREFIX.encode("utf-8"), b"/")
        if isinstance(content, str):
            return content.replace(_INTERNAL_FLOW_PATH_PREFIX, "/")
        return content
