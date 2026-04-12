import base64
import json
import argparse
import asyncio
import os
import shutil
import tempfile
import unittest

from snare.tanner_handler import TannerHandler
from snare.utils.asyncmock import AsyncMock


class TestParseSeedEndpoints(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="snare-seed-test-")
        run_args = argparse.ArgumentParser()
        run_args.add_argument("--tanner")
        run_args.add_argument("--page-dir")
        args = run_args.parse_args(["--page-dir", "test-pages"])
        args_dict = vars(args)
        args_dict["full_page_path"] = self.temp_dir
        args.no_dorks = True
        args.index_page = "/index.html"
        args.tanner = "tanner.local"

        self.handler = TannerHandler(args, {}, b"9c10172f-7ce2-4fb4-b1c6-abc70141db56")
        self.seed_file_path = os.path.join(self.temp_dir, "seed-endpoints.txt")
        self.loop = asyncio.new_event_loop()

    def test_parse_seed_endpoints_normalizes_and_deduplicates(self):
        with open(self.seed_file_path, "w") as seed_file:
            seed_file.write("/admin\n")
            seed_file.write("admin\n")
            seed_file.write("/admin?source=seed\n")
            seed_file.write("http://ip-api.com/json/\n")
            seed_file.write("HTTPS://example.org/health\n")
            seed_file.write("/\n")
            seed_file.write("/index.html\n")
            seed_file.write("#ignored\n")
            seed_file.write("\n")

        parsed_endpoints = self.handler.parse_seed_endpoints(self.seed_file_path)

        self.assertEqual(parsed_endpoints, ["/admin", "/index.html"])

    def test_parse_seed_endpoints_empty_path_returns_empty_list(self):
        self.assertEqual(self.handler.parse_seed_endpoints(None), [])

    def test_consume_seed_endpoints_skips_existing_and_generates_missing(self):
        self.handler.meta["/existing"] = {"hash": "abc", "headers": []}
        self.handler._request_meta_generate_job = AsyncMock(return_value="job-1")
        self.handler.poll_meta_job = AsyncMock(return_value=True)

        summary = self.loop.run_until_complete(
            self.handler.consume_seed_endpoints(["/existing", "/missing"]),
        )

        self.assertEqual(summary, {"requested": 2, "skipped": 1, "generated": 1, "failed": 0})
        self.handler._request_meta_generate_job.assert_called_once_with("/missing")
        self.handler.poll_meta_job.assert_called_once_with("job-1", "/missing")

    def test_consume_seed_endpoints_counts_missing_job_id_as_failed(self):
        self.handler._request_meta_generate_job = AsyncMock(return_value=None)
        self.handler.poll_meta_job = AsyncMock(return_value=True)

        summary = self.loop.run_until_complete(self.handler.consume_seed_endpoints(["/missing"]))

        self.assertEqual(summary, {"requested": 1, "skipped": 0, "generated": 0, "failed": 1})
        self.handler.poll_meta_job.assert_not_called()

    def test_save_generated_artifacts_persists_bundle_paths(self):
        artifacts = [
            {
                "path": "/missing",
                "headers": [{"Content-Type": "text/plain; charset=utf-8"}],
                "body_b64": base64.b64encode(b"secret=value\n").decode("ascii"),
                "status_code": 200,
            },
            {
                "path": "/robots.txt",
                "headers": [{"Content-Type": "text/plain; charset=utf-8"}],
                "body_b64": base64.b64encode(b"User-agent: *\nDisallow: /private\n").decode("ascii"),
                "status_code": 200,
            },
        ]

        self.loop.run_until_complete(self.handler._save_generated_artifacts(artifacts, "/missing"))

        self.assertIn("/missing", self.handler.meta)
        self.assertIn("/robots.txt", self.handler.meta)
        with open(os.path.join(self.temp_dir, "meta.json")) as meta_file:
            persisted_meta = json.load(meta_file)
        self.assertIn("/missing", persisted_meta)
        self.assertIn("/robots.txt", persisted_meta)
        with open(os.path.join(self.temp_dir, self.handler.meta["/robots.txt"]["hash"]), "rb") as content_file:
            self.assertEqual(content_file.read(), b"User-agent: *\nDisallow: /private\n")


    def tearDown(self):
        self.loop.close()
        shutil.rmtree(self.temp_dir)