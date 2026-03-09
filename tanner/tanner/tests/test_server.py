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
