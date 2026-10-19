#!/usr/bin/env python3
"""Compatibility wrapper for scripts/habitat_humanoid_bridge.py.

The implementation lives in scripts/collection/habitat_humanoid_bridge.py. This wrapper keeps old
commands such as `python scripts/habitat_humanoid_bridge.py` working after the script cleanup.
"""
from pathlib import Path
import importlib.util
import runpy
import sys

_HERE = Path(__file__).resolve().parent
_IMPL = _HERE / "collection" / "habitat_humanoid_bridge.py"
for _category in ['collection', 'preprocess', 'training', 'evaluation', 'debug', 'legacy']:
    _path = str(_HERE / _category)
    if _path not in sys.path:
        sys.path.insert(0, _path)

if __name__ == "__main__":
    runpy.run_path(str(_IMPL), run_name="__main__")
else:
    _spec = importlib.util.spec_from_file_location(f"_moved_{_IMPL.stem}", _IMPL)
    _module = importlib.util.module_from_spec(_spec)
    assert _spec.loader is not None
    sys.modules[_spec.name] = _module
    _spec.loader.exec_module(_module)
    globals().update({
        _key: _value
        for _key, _value in vars(_module).items()
        if not (_key.startswith("__") and _key not in {"__all__", "__doc__"})
    })
