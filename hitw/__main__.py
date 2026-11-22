"""The hitw command line: ``python3 hitw <command>`` from the repository root
(or ``python3 path/to/hitw <command>`` from anywhere), or ``python3 -m hitw
<command>`` with the repository on the import path.

Run as a directory, hitw is not imported as a package: this file puts the
repository, not ``hitw/`` itself, on the import path (otherwise hitw's
modules, such as config.py, would shadow top-level imports).  The
interpreter must have the dependencies listed in README.md.
"""

import sys
from pathlib import Path


def _bootstrap() -> None:
    """Make ``python3 hitw ...`` behave like ``python3 -m hitw ...`` run from the repository."""
    here = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != here]
    sys.path.insert(0, str(here.parent))


if __name__ == "__main__":
    if not __package__:
        _bootstrap()
    try:
        import numpy, pydrake, scipy, shapely, pandas, matplotlib  # noqa: E401, F401
    except ImportError as e:
        sys.exit(f"hitw: {e.name} is not installed for {sys.executable}; install the dependencies "
                 f"(README.md, 'Environment') and run hitw with that environment's python3")
    from hitw.cli import main
    sys.exit(main())
