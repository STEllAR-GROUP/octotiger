"""Mechanical Step 03 checks: legacy and canonical launchers agree."""
from pathlib import Path
import json
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class MigrationTests(unittest.TestCase):
    def test_old_and_new_wrappers_compare_results_and_diagnostics(self):
        result_root = ROOT / "verification_results" / "results"
        result_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=result_root) as work:
            work = Path(work)
            fake = work / "fake_runner.py"
            fake.write_text("""#!/usr/bin/env python3
import json, pathlib, sys
a = sys.argv[1:]
out = pathlib.Path(a[a.index('--output') + 1])
out.mkdir(parents=True, exist_ok=True)
(out / 'result.dat').write_bytes(b'case=streaming_wave\\nL1=0\\n')
meta = {'diagnostics': {'L1': 0.0, 'L2': 0.0}, 'parameters': {'levels': [2, 3], 'build': 'Release'}}
(out / 'run.json').write_text(json.dumps(meta, sort_keys=True) + '\\n')
(out / 'batch.json').write_text(json.dumps({'diagnostics': meta['diagnostics']}, sort_keys=True) + '\\n')
""", encoding="utf-8")
            fake.chmod(0o755)
            old_out, new_out = work / "old", work / "new"
            env = dict(os.environ, OCTOTIGER_VERIFICATION_RADIATION_RUNNER=str(fake))
            common = ["streaming_wave", "2", "3", "Release", "--output"]
            subprocess.run([str(ROOT / "radiation_results/run.sh"), *common, str(old_out)],
                           cwd=ROOT, env=env, check=True)
            subprocess.run([str(ROOT / "verification_results/run.sh"), "run",
                            "radiation.skinner_ostriker.streaming_wave", "2", "3",
                            "Release", "--output", str(new_out)],
                           cwd=ROOT, env=env, check=True)
            self.assertEqual((old_out / "result.dat").read_bytes(),
                             (new_out / "result.dat").read_bytes())
            for name in ("run.json", "batch.json"):
                self.assertEqual(json.loads((old_out / name).read_text()),
                                 json.loads((new_out / name).read_text()))


if __name__ == "__main__":
    unittest.main()
