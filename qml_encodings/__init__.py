# encodings/__init__.py
from importlib import import_module
from types import ModuleType
from typing import Any

def _get_qml_encodings(mod_name: str) -> Any:
	mod = import_module(f".{mod_name}", package=__package__)
	# prefer the attribute qml_encodings if present, otherwise return the module itself
	return getattr(mod, "qml_encodings", mod)

angle_xy = _get_qml_encodings("angle_xy")
iqp = _get_qml_encodings("iqp")
basis = _get_qml_encodings("basis")
bphe = _get_qml_encodings("bphe")
amplitude = _get_qml_encodings("amplitude")