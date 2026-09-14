"""Mock social media platform package.

NAME-COLLISION NOTE (deliberate and documented — the architecture doc's
section-3 layout specifies this exact folder name, which shadows Python's
stdlib `platform` module):

Library code that does a bare `import platform` — notably SQLAlchemy's compat
layer, which calls platform.python_implementation() — would otherwise receive
THIS package and crash with AttributeError. Renaming the folder would deviate
from the doc's structure, so instead this package re-exports the stdlib
module's public names at import time: `platform.python_implementation()` keeps
working, while `from platform.db import ...` resolves to our code.

The stdlib file is loaded BY PATH (it is shadowed on sys.path, so a normal
import would find us — the very problem we're fixing).

Verified by tests/test_simulation_engine.py::test_platform_package_shadows_stdlib_safely.
"""

import importlib.util
import sys
import sysconfig
from pathlib import Path


def _load_stdlib_platform():
    """Load the real stdlib platform.py under a private name."""
    stdlib_dir = Path(sysconfig.get_paths()["stdlib"])
    source = stdlib_dir / "platform.py"
    spec = importlib.util.spec_from_file_location("_stdlib_platform_shim", source)
    if spec is None or spec.loader is None:  # pragma: no cover - stdlib always present
        raise ImportError(f"could not locate stdlib platform.py at {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_stdlib = _load_stdlib_platform()

# Re-export every public stdlib name (python_implementation, system, release,
# machine, node, processor, ...) into this package's namespace.
for _name in dir(_stdlib):
    if not _name.startswith("_"):
        globals().setdefault(_name, getattr(_stdlib, _name))

# Submodules (db, routes, simulation, main) are NOT imported here on purpose:
# importing them at package-init time would run their `import sqlalchemy`
# BEFORE the re-exports above complete, recreating the original crash as a
# circular import. Application code imports them explicitly
# (`from platform.db import ...`) which works fine after this init finishes.
