"""Minimal test runner (no pytest): imports every tests/test_*.py and runs
each `test_*` function, reporting pass/fail and timing.

Run:  .venv/bin/python tests/run_tests.py [substring filter]
"""

import importlib.util
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv):
    pattern = argv[1] if len(argv) > 1 else ""
    failures = 0
    total = 0
    for path in sorted(Path(__file__).parent.glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for name in dir(mod):
            if not name.startswith("test_") or pattern not in f"{path.stem}.{name}":
                continue
            total += 1
            t0 = time.perf_counter()
            try:
                getattr(mod, name)()
                print(f"PASS {path.stem}.{name} ({time.perf_counter() - t0:.2f}s)")
            except Exception:
                failures += 1
                print(f"FAIL {path.stem}.{name}")
                traceback.print_exc()
    print(f"\n{total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
