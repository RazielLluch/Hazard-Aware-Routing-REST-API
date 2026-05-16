"""Vendored Macro-DDQN runner and the FastAPI bridge into it.

``runner.py`` is a near-verbatim copy of the upstream Macro-DDQN training /
evaluation script with a single vendor patch: ``REPO_ROOT`` is made configurable
via the ``MACRO_DDQN_ROOT`` environment variable so the runner resolves the
hazard graph, configs, and artefacts from the standalone vendor tree at
``data/macro_vendor/`` (mirroring the canonical layout). ``bridge.py`` sets the
env var from ``settings.macro_vendor_root`` at startup. See ``bridge.py`` for
the API-facing surface.
"""
