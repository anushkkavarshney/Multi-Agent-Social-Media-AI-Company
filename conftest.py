"""Pytest bootstrap (repo root — runs before test collection).

pytest imports the stdlib `platform` module during its own startup, so by the
time test modules load, sys.modules['platform'] is the stdlib module and any
`from platform.simulation import ...` would fail ("platform is not a
package"). We rebind sys.modules['platform'] to the LOCAL package here.

Safety: code that imported the stdlib module before this rebind (pytest
internals) still holds a direct reference and keeps working. Code that imports
`platform` afterwards gets our package, which re-exports the stdlib names it
needs (see platform/__init__.py) — so `platform.python_implementation()` from
library code keeps working either way.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Drop the cached stdlib module so the re-import below resolves to OUR package
# (ROOT is now first on sys.path).
sys.modules.pop("platform", None)
import platform as _local_platform  # noqa: E402

assert hasattr(_local_platform, "__path__"), (
    "conftest failed: sys.modules['platform'] is not the local package — "
    "tests cannot import platform.* submodules"
)
