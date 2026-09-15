"""Lit config for the Estela compiler tests.

These tests live outside the IREE source tree, so lit will not find
3rdparty/iree/compiler/lit.cfg.py by walking upwards from them; this file is
the config for this directory. It is deliberately a near-copy of IREE's own so
the two behave the same way.
"""

# config is injected by lit, not defined here.
# pylint: disable=undefined-variable

import os
import tempfile

import lit.formats

config.name = "Estela"
config.suffixes = [".mlir"]
config.test_format = lit.formats.ShTest(
    execute_external=True, force_execute_external=True
)

# Forward IREE_* environment variables so the tools behave the same as when run
# by hand.
config.environment.update(
    {k: v for k, v in os.environ.items() if k.startswith("IREE_")}
)

config.test_exec_root = (
    os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    or os.environ.get("TEST_TMPDIR")
    or os.path.join(tempfile.gettempdir(), "lit")
)
