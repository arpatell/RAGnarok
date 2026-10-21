"""Exercise lifecycle scripts with fake systemd/network commands, never real services."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
BASH = shutil.which("bash") if os.name != "nt" else "C:/Program Files/Git/bin/bash.exe"


def shell_path(path):
    value = str(path).replace("\\", "/")
    return f"/{value[0].lower()}{value[2:]}" if len(value) > 1 and value[1] == ":" else value


class ServiceLifecycleTests(unittest.TestCase):
    def setUp(self):
        (ROOT / ".tmp").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".tmp")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.trace = self.directory / "trace"
        self.bin = self.directory / "bin"
        self.bin.mkdir()
        self.stub("flock", 'exit "${LOCK_RESULT:-0}"')
        self.stub("sudo", 'exec "$@"')
        self.stub("sleep", "exit 0")
        self.stub("python3", 'exec "$TEST_PYTHON" "$@"')
        self.stub("journalctl", 'echo "fresh diagnostic output"')
        self.stub("systemctl", '''
printf '%s\\n' "systemctl $*" >> "$TRACE"
case "$1" in
  is-active)
    if [ "${MODE:-}" = failed ]; then echo failed; exit 3; fi
    if [ "${MODE:-}" = transition ] && [ ! -f "$STATE/transition" ]; then
      touch "$STATE/transition"; echo deactivating; exit 3
    fi
    echo active;;
  show)
    case "$*" in
      *ExecMainStartTimestamp*)
        if [ "${MODE:-}" = young ]; then date --iso-8601=seconds; else echo '2000-01-01 00:00:00 UTC'; fi;;
      *) echo 'ActiveState=failed';;
    esac;;
  restart)
    touch "$STATE/restarted"
    if [ "${MODE:-}" = restart_failure ]; then exit 1; fi;;
esac
''')
        self.stub("curl", '''
printf '%s\\n' "curl $*" >> "$TRACE"
if [ "${MODE:-}" = recover ] && [ ! -f "$STATE/restarted" ]; then exit 7; fi
if [ "${MODE:-}" = pending ] && [ ! -f "$STATE/probed" ]; then
  touch "$STATE/probed"; echo '{"initialized":false}'; exit 0
fi
case "$*" in
  *8090*) echo '{"status":"ok","initialized":"true"}';;
  *8787*) echo '{"ok":true}';;
  *) exit 1;;
esac
''')

    def stub(self, name, body):
        path = self.bin / name
        path.write_text("#!/usr/bin/env bash\n" + body + "\n", newline="\n")
        path.chmod(0o755)

    def run_script(self, name, **updates):
        env = {
            **os.environ,
            "PATH": shell_path(self.bin) + ":/usr/bin:/bin",
            "MOCK_BIN": shell_path(self.bin),
            "TEST_PYTHON": sys.executable.replace("\\", "/"),
            "TRACE": shell_path(self.trace), "STATE": shell_path(self.directory),
            "RAG_WATCHDOG_LOCK_FILE": shell_path(self.directory / "lock"),
            "RAG_WATCHDOG_LOG_FILE": shell_path(self.directory / "watchdog.log"),
            **updates,
        }
        result = subprocess.run([
            BASH, "-c", 'export PATH="$MOCK_BIN:/usr/bin:/bin"; exec /usr/bin/bash "$1"',
            "lifecycle-test", str(ROOT / "scripts/gcp" / name)
        ], env=env, capture_output=True, text=True, timeout=15)
        trace = self.trace.read_text() if self.trace.exists() else ""
        return result, trace

    def test_deploy_restarts_backend_only_after_rag_readiness(self):
        result, trace = self.run_script("restart-services.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertLess(trace.index("8090/health"), trace.index("restart ragnarok-backend"))
        self.assertIn("8787/health", trace)

    def test_transient_deactivation_is_not_treated_as_a_crash(self):
        result, _ = self.run_script("restart-services.sh", MODE="transition")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_health_must_report_initialized_not_merely_http_200(self):
        result, trace = self.run_script("restart-services.sh", MODE="pending")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(trace.count("8090/health"), 2)

    def test_failed_rag_does_not_restart_backend(self):
        result, trace = self.run_script("restart-services.sh", MODE="failed")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("restart ragnarok-backend", trace)
        self.assertIn("fresh diagnostic output", result.stdout)

    def test_restart_command_failure_prints_diagnostics(self):
        result, trace = self.run_script("restart-services.sh", MODE="restart_failure")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("restart ragnarok-backend", trace)
        self.assertIn("fresh diagnostic output", result.stdout)

    def test_deploy_times_out_without_false_success(self):
        result, trace = self.run_script("restart-services.sh", RAG_STARTUP_TIMEOUT_SECONDS="0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Timed out", result.stdout)
        self.assertNotIn("restart ragnarok-backend", trace)

    def test_deploy_with_busy_lock_does_not_touch_services(self):
        result, trace = self.run_script("restart-services.sh", LOCK_RESULT="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(trace, "")

    def test_watchdog_busy_lock_exits_without_probes(self):
        result, trace = self.run_script("rag-watchdog.sh", LOCK_RESULT="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(trace, "")

    def test_watchdog_leaves_cold_start_alone(self):
        result, trace = self.run_script("rag-watchdog.sh", MODE="young")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("curl", trace)
        self.assertNotIn("restart", trace)

    def test_watchdog_probes_health_without_search_or_model_calls(self):
        result, trace = self.run_script("rag-watchdog.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("8090/health", trace)
        self.assertNotIn("/rag/search", trace)
        self.assertNotIn("restart", trace)

    def test_watchdog_still_recovers_an_unhealthy_old_process(self):
        result, trace = self.run_script("rag-watchdog.sh", MODE="recover")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(trace.count("restart ragnarok-rag.service"), 1)


if __name__ == "__main__":
    unittest.main()
