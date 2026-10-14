#!/usr/bin/env python
"""``antiGen-train`` with a workaround for this environment's broken torchvision.

``torchvision 0.21.0`` is built for torch 2.6 and raises
``operator torchvision::nms does not exist`` under the installed torch 2.7.1. transformers
4.57 imports ``torchvision.transforms`` on the way to ``EsmForMaskedLM`` — guarded by
``is_torchvision_available()`` — so ``--model-type esm2`` fails with a misleading
``ModuleNotFoundError: Could not import module 'EsmForMaskedLM'``. Telling transformers
that torchvision is unavailable, before any of its other submodules are imported, restores
the ESM-2 path. ESM-2 itself needs nothing from torchvision.

Delete this file once the environment is fixed properly, with either

    pip install 'torchvision==0.22.1'      # the build matching torch 2.7.1
    pip uninstall torchvision             # nothing in antiGen uses it

and call ``antiGen-train`` directly. Every argument here is passed straight through.
"""
import sys

import transformers.utils.import_utils as _import_utils

_import_utils._torchvision_available = False
_import_utils._torchvision_version = "N/A"

from antiGen.train import main  # noqa: E402  (must follow the patch above)

if __name__ == "__main__":
    sys.exit(main())
