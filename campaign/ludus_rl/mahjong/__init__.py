"""Ludus MCR (Chinese Standard Mahjong) package."""

# Exposed so the IJCAI validator's ``__import__("mahjong.validate_adapter")``
# (which returns the top-level ``mahjong`` package) can reach the engine class
# via ``getattr(pkg, "MyEngine")``.
from mahjong.validate_adapter import MyEngine  # noqa: F401
