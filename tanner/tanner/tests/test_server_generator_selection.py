import sys
import types
import unittest
from unittest import mock

from tanner.generator import base_generator

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


class TestServerGeneratorSelection(unittest.TestCase):
    @staticmethod
    def _config_get_without_generator(section, value):
        values = {
            ("EMULATORS", "root_dir"): "/opt/tanner",
            ("SQLI", "db_name"): "tanner_db",
            ("SESSIONS", "delete_timeout"): 300,
            ("HPFEEDS", "enabled"): False,
        }
        return values.get((section, value))

    @staticmethod
    def _config_get_local_qwen(section, value):
        values = {
            ("EMULATORS", "root_dir"): "/opt/tanner",
            ("SQLI", "db_name"): "tanner_db",
            ("SESSIONS", "delete_timeout"): 300,
            ("HPFEEDS", "enabled"): False,
            ("GENERATOR", "backend"): "local_qwen",
        }
        return values.get((section, value))

    def test_default_backend_uses_base_generator(self):
        with mock.patch("tanner.server.TannerConfig.get", side_effect=self._config_get_without_generator):
            with mock.patch("tanner.dorks_manager.DorksManager", mock.Mock()):
                with mock.patch("tanner.emulators.base.BaseHandler", mock.Mock(), create=True):
                    with mock.patch("tanner.sessions.session_manager.SessionManager", mock.Mock(), create=True):
                        serv = server.TannerServer()

        self.assertIsInstance(serv.generator, base_generator.BaseGenerator)

    def test_local_qwen_backend_uses_local_generator(self):
        with mock.patch("tanner.server.TannerConfig.get", side_effect=self._config_get_local_qwen):
            with mock.patch("tanner.dorks_manager.DorksManager", mock.Mock()):
                with mock.patch("tanner.emulators.base.BaseHandler", mock.Mock(), create=True):
                    with mock.patch("tanner.sessions.session_manager.SessionManager", mock.Mock(), create=True):
                        with mock.patch("tanner.server.local_qwen_generator.LocalQwenGenerator") as generator_cls:
                            generator_instance = mock.Mock()
                            generator_cls.return_value = generator_instance
                            serv = server.TannerServer()

        self.assertIs(serv.generator, generator_instance)
        generator_cls.assert_called_once_with()
