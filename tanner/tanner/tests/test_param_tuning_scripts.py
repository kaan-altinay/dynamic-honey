import importlib.util
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
PARAM_TUNING_DIR = REPO_ROOT / "param_tuning"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestParamTuningScripts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix_module = _load_module(
            "test_run_param_matrix_module",
            PARAM_TUNING_DIR / "run_param_matrix.py",
        )
        cls.tuning_module = _load_module(
            "test_run_param_tuning_module",
            PARAM_TUNING_DIR / "run_param_tuning.py",
        )

    def test_resume_results_restart_first_incomplete_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            params_dir = Path(tmp_dir)
            completed_endpoint = "/boaform/admin/formLogin"
            incomplete_endpoint = "/wp-login.php"
            endpoints = [
                completed_endpoint,
                incomplete_endpoint,
                "/api/v1/pods",
            ]
            completed_dir = params_dir / self.tuning_module._slugify_endpoint(completed_endpoint)
            completed_dir.mkdir(parents=True)
            incomplete_dir = params_dir / self.tuning_module._slugify_endpoint(incomplete_endpoint)
            incomplete_dir.mkdir(parents=True)
            (incomplete_dir / "partial.txt").write_text("incomplete")
            self.tuning_module._write_json(
                self.tuning_module._progress_path(params_dir),
                {
                    "results": [
                        {
                            "endpoint": completed_endpoint,
                            "folder": completed_dir.name,
                            "primary_path": completed_endpoint,
                        }
                    ],
                    "complete": False,
                },
            )

            results, pending_endpoints, is_complete = self.tuning_module._resume_results_and_pending_endpoints(
                params_dir,
                endpoints,
            )

            self.assertEqual([summary["endpoint"] for summary in results], [completed_endpoint])
            self.assertEqual(pending_endpoints, endpoints[1:])
            self.assertFalse(is_complete)
            self.assertTrue(completed_dir.exists())
            self.assertFalse(incomplete_dir.exists())

    def test_run_plan_resume_skips_completed_params_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = Path(tmp_dir)
            plan = self.matrix_module.build_plan(["A"])
            entry_complete = plan[0]
            entry_pending = plan[1]
            run_name = "run2"
            endpoints = list(self.matrix_module.DEFAULT_ENDPOINTS)

            complete_dir = self.matrix_module._family_run_dir(run_root, entry_complete["family"], run_name) / entry_complete["params_slug"]
            complete_dir.mkdir(parents=True, exist_ok=True)
            (complete_dir / "run_state.json").write_text(
                self.matrix_module.json.dumps(
                    {
                        "complete": True,
                        "results": [
                            {"endpoint": endpoint}
                            for endpoint in endpoints
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )

            args = types.SimpleNamespace(
                model_name="gpt-5.4",
                page_url="example.com",
                device_endpoint="/wsman",
                run_root=str(run_root),
                families="A",
                resume=True,
            )

            with mock.patch.object(self.matrix_module.subprocess, "run") as mock_run:
                self.matrix_module.run_plan(plan[:2], args, run_name)

            mock_run.assert_called_once()
            cmd = mock_run.call_args.kwargs["args"] if "args" in mock_run.call_args.kwargs else mock_run.call_args.args[0]
            self.assertIn("--resume", cmd)
            self.assertIn("--family-run-dir", cmd)
            self.assertIn(str(self.matrix_module._family_run_dir(run_root, entry_pending["family"], run_name)), cmd)
            self.assertIn("--params-slug", cmd)
            self.assertIn(entry_pending["params_slug"], cmd)


if __name__ == "__main__":
    unittest.main()
