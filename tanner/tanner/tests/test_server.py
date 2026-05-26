import base64
import sys
import types
import uuid
from unittest import mock
import hashlib

from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop

if "aioredis" not in sys.modules:
    fake_aioredis = types.ModuleType("aioredis")
    fake_aioredis.from_url = lambda *args, **kwargs: None
    fake_aioredis.exceptions = types.SimpleNamespace(ConnectionError=Exception)
    sys.modules["aioredis"] = fake_aioredis

if "redis" not in sys.modules:
    fake_redis = types.ModuleType("redis")
    fake_redis.asyncio = types.SimpleNamespace()
    sys.modules["redis"] = fake_redis

if "aiodocker" not in sys.modules:
    fake_aiodocker = types.ModuleType("aiodocker")
    fake_aiodocker.Docker = lambda *args, **kwargs: mock.Mock()
    fake_aiodocker.exceptions = types.SimpleNamespace(DockerError=Exception, DockerContainerError=Exception)
    sys.modules["aiodocker"] = fake_aiodocker

if "aiomysql" not in sys.modules:
    fake_aiomysql = types.ModuleType("aiomysql")
    async def _fake_connect(*args, **kwargs):
        return mock.AsyncMock()
    fake_aiomysql.connect = _fake_connect
    sys.modules["aiomysql"] = fake_aiomysql

if "mimesis" not in sys.modules:
    fake_mimesis = types.ModuleType("mimesis")
    class _FakePerson:
        def __init__(self, *args, **kwargs):
            pass
        def username(self):
            return "user"
        def email(self):
            return "user@example.com"
        def password(self):
            return "password"
    class _FakeText:
        def text(self, quantity=1):
            return "text"
    fake_mimesis.Person = _FakePerson
    fake_mimesis.Text = _FakeText
    sys.modules["mimesis"] = fake_mimesis

if "geoip2" not in sys.modules:
    fake_geoip2 = types.ModuleType("geoip2")
    fake_geoip2_database = types.ModuleType("geoip2.database")
    class _FakeReader:
        def __init__(self, *args, **kwargs):
            pass
    fake_geoip2_database.Reader = _FakeReader
    fake_geoip2.database = fake_geoip2_database
    sys.modules["geoip2"] = fake_geoip2
    sys.modules["geoip2.database"] = fake_geoip2_database

if "bson" not in sys.modules:
    fake_bson = types.ModuleType("bson")
    fake_bson_objectid = types.ModuleType("bson.objectid")
    fake_bson_objectid.ObjectId = lambda value=None: value or "object-id"
    fake_bson.objectid = fake_bson_objectid
    sys.modules["bson"] = fake_bson
    sys.modules["bson.objectid"] = fake_bson_objectid

if "gridfs" not in sys.modules:
    fake_gridfs = types.ModuleType("gridfs")
    fake_gridfs.GridFS = lambda *args, **kwargs: mock.Mock()
    sys.modules["gridfs"] = fake_gridfs

if "pylibinjection" not in sys.modules:
    fake_pylibinjection = types.ModuleType("pylibinjection")
    fake_pylibinjection.sqli = lambda value, flags=0: (False, "")
    fake_pylibinjection.xss = lambda value, flags=0: False
    sys.modules["pylibinjection"] = fake_pylibinjection

from tanner import server
from tanner.config import TannerConfig
from tanner.utils.asyncmock import AsyncMock
from tanner import __version__ as tanner_version


class TestServer(AioHTTPTestCase):
    def setUp(self):
        d = dict(
            MONGO={"enabled": "False", "URI": "mongodb://localhost"},
            LOCALLOG={"enabled": "False", "PATH": "/tmp/tanner_report.json"},
        )
        m = mock.MagicMock()
        m.__getitem__.side_effect = d.__getitem__
        m.__iter__.side_effect = d.__iter__

        with mock.patch("tanner.tests.test_server.TannerConfig") as p:

            TannerConfig.config = m
            TannerConfig.get = m.get

        with mock.patch("tanner.dorks_manager.DorksManager", mock.Mock()):
            with mock.patch("tanner.emulators.base.BaseHandler", mock.Mock(), create=True):
                with mock.patch("tanner.sessions.session_manager.SessionManager", mock.Mock(), create=True):
                    self.serv = server.TannerServer()

        self.test_uuid = uuid.uuid4()

        async def _add_or_update_mock(data, client):
            sess = mock.Mock()
            sess.set_attack_type = mock.Mock()
            sess_id = hashlib.md5(b"foo")
            test_uuid = uuid
            sess.get_uuid = mock.Mock(return_value=str(self.test_uuid))
            return sess, sess_id

        async def _delete_sessions_mock(client):
            pass

        self.serv.session_manager.add_or_update_session = _add_or_update_mock
        self.serv.session_manager.delete_sessions_on_shutdown = _delete_sessions_mock
        self.serv.session_manager.delete_old_sessions = _delete_sessions_mock

        async def choosed(client):
            return [x for x in range(10)]

        dorks = mock.Mock()
        dorks.choose_dorks = choosed
        dorks.extract_path = self._make_coroutine()

        redis = AsyncMock()
        redis.close = AsyncMock()
        self.serv.dorks = dorks
        self.serv.redis_client = redis

        super(TestServer, self).setUp()

    def _make_coroutine(self):
        async def coroutine(*args, **kwargs):
            return mock.Mock(*args, **kwargs)

        return coroutine

    async def get_application(self):
        app = await self.serv.make_app()
        return app

    @unittest_run_loop
    async def test_example(self):
        request = await self.client.request("GET", "/")
        assert request.status == 200
        text = await request.text()
        assert "Tanner server" in text

    def test_meta_policy_blocks_ua_only_signal(self):
        policy = server.MetaGenerationPolicy(mock.Mock())
        allowed, reason = policy.should_generate(
            path="/probe",
            method="GET",
            src_ip="10.0.0.1",
            user_agent="CensysInspect/1.1",
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "ua_only_positive")

    def test_meta_policy_allows_high_value_signal(self):
        policy = server.MetaGenerationPolicy(mock.Mock())
        allowed, reason = policy.should_generate(
            path="/api/version",
            method="GET",
            src_ip="10.0.0.2",
            user_agent="",
        )
        self.assertTrue(allowed)
        self.assertEqual(reason, "scheduled")

    def test_make_response(self):
        msg = "test"
        content = self.serv._make_response(msg)
        assert_content = dict(version=tanner_version, response=dict(message=msg))
        self.assertDictEqual(content, assert_content)

    @unittest_run_loop
    async def test_events_request(self):
        async def _make_handle_coroutine(*args, **kwargs):
            return {"name": "index", "order": 1, "payload": None}

        detection_assert = {
            "version": tanner_version,
            "response": {
                "message": {
                    "detection": {"name": "index", "order": 1, "payload": None},
                    "sess_uuid": str(self.test_uuid),
                }
            },
        }
        self.serv.base_handler.handle = _make_handle_coroutine
        request = await self.client.request("POST", "/event", data=b'{"path":"/index.html"}')
        assert request.status == 200
        detection = await request.json()
        self.assertDictEqual(detection, detection_assert)


    @unittest_run_loop
    async def test_meta_generate_request(self):
        self.serv._save_meta_job = AsyncMock()
        self.serv._run_meta_job = AsyncMock()

        with mock.patch("tanner.server.asyncio.create_task") as create_task:
            create_task.side_effect = lambda coro: coro.close()
            request = await self.client.request(
                "POST",
                "/meta_generate",
                data=b'{"path":"/seed/page","host":"seed.example","index_page":"/index.html"}',
            )

        assert request.status == 202
        response = await request.json()
        message = response["response"]["message"]
        self.assertEqual(message["state"], "pending")
        self.assertEqual(message["path"], "/seed/page")
        self.assertIn("meta_job_id", message)

        self.serv._save_meta_job.assert_called_once()
        save_job_args = self.serv._save_meta_job.call_args[0]
        self.assertEqual(save_job_args[1]["state"], "pending")
        self.assertEqual(save_job_args[1]["path"], "/seed/page")
        create_task.assert_called_once()

    @unittest_run_loop
    async def test_meta_generate_invalid_payload(self):
        request = await self.client.request("POST", "/meta_generate", data=b'{}')
        assert request.status == 400
        response = await request.json()
        self.assertEqual(response["response"]["message"], "KeyError")
    @unittest_run_loop
    async def test_run_meta_job_saves_bundle_payload(self):
        bundle = {
            "primary_path": "/seed/page",
            "artifacts": [
                {
                    "path": "/seed/page",
                    "kind": "html_page",
                    "headers": [{"Content-Type": "text/html; charset=utf-8"}],
                    "body_bytes": b"<html><body>seed</body></html>",
                    "status_code": 200,
                    "source_artifact_id": "primary-page",
                    "artifact_scope": "static_file",
                },
                {
                    "path": "/robots.txt",
                    "kind": "robots_txt",
                    "headers": [{"Content-Type": "text/plain; charset=utf-8"}],
                    "body_bytes": b"User-agent: *\nDisallow: /private\n",
                    "status_code": 200,
                    "source_artifact_id": "robots",
                    "artifact_scope": "static_file",
                },
            ],
            "review_summary": "approved",
            "used_fallback": False,
            "flow_descriptor": {
                "rules": [
                    {
                        "match_path": "/seed/page",
                        "condition": {"missing_header": "Authorization"},
                        "response": {
                            "artifact_path": "/_flow/seed-page/auth-required",
                            "status_code": 401,
                        },
                        "priority": 10,
                    }
                ]
            },
        }
        self.serv.generator.generate_bundle = AsyncMock(return_value=bundle)
        self.serv._save_meta_job = AsyncMock()

        await self.serv._run_meta_job("job-1", "seed.example", "/seed/page", {"index_page": "/index.html"})

        self.serv._save_meta_job.assert_called_once()
        saved_fields = self.serv._save_meta_job.call_args[0][1]
        self.assertEqual(saved_fields["state"], "ready")
        self.assertEqual(saved_fields["primary_path"], "/seed/page")
        self.assertEqual(saved_fields["review_summary"], "approved")
        self.assertEqual(len(saved_fields["artifacts"]), 2)
        self.assertIn("flow_descriptor", saved_fields)
        self.assertEqual(saved_fields["flow_descriptor"]["rules"][0]["match_path"], "/seed/page")
        self.assertEqual(saved_fields["artifacts"][0]["path"], "/seed/page")
        self.assertEqual(
            base64.b64decode(saved_fields["artifacts"][0]["body_b64"]),
            b"<html><body>seed</body></html>",
        )

    @unittest_run_loop
    async def test_meta_job_ready_returns_bundle_payload(self):
        self.serv.redis_client.hgetall = AsyncMock(
            return_value={
                "state": "ready",
                "primary_path": "/seed/page",
                "artifacts": '[{"path": "/seed/page", "kind": "html_page", "headers": [{"Content-Type": "text/html; charset=utf-8"}], "body_b64": "c2VlZA==", "status_code": 200}]',
                "review_summary": "approved",
                "used_fallback": "False",
                "flow_descriptor": '{"rules": [{"match_path": "/seed/page", "condition": {"missing_header": "Authorization"}, "response": {"artifact_path": "/_flow/seed-page/auth-required", "status_code": 401}, "priority": 10}]}',
            }
        )

        request = await self.client.request("GET", "/meta_job/job-123")

        assert request.status == 200
        response = await request.json()
        message = response["response"]["message"]
        self.assertEqual(message["state"], "ready")
        self.assertEqual(message["primary_path"], "/seed/page")
        self.assertEqual(message["review_summary"], "approved")
        self.assertFalse(message["used_fallback"])
        self.assertEqual(len(message["artifacts"]), 1)
        self.assertEqual(message["artifacts"][0]["path"], "/seed/page")
        self.assertEqual(message["flow_descriptor"]["rules"][0]["match_path"], "/seed/page")


    @unittest_run_loop
    async def test_dorks_request(self):
        assert_content = dict(version=tanner_version, response=dict(dorks=[x for x in range(10)]))
        request = await self.client.request("GET", "/dorks")
        assert request.status == 200
        detection = await request.json()
        self.assertDictEqual(detection, assert_content)

    @unittest_run_loop
    async def test_version(self):
        assert_content = dict(version=tanner_version)
        request = await self.client.request("GET", "/version")
        assert request.status == 200
        detection = await request.json()
        self.assertDictEqual(detection, assert_content)
