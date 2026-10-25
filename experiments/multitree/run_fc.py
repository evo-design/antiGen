#!/usr/bin/env python
"""Entry point for the forecast experiment, using the IN-REPO vendored antiGen under
Ivan's home checkout.

The vendored train.py has a real `--select-metric {nll_pooled,recall10_macro}` flag and
the vendored eval_utils masks gap/stop columns, so no runtime patching is needed. The
sys.path.insert wins over the ag_env editable finder (verified), so `import antiGen`
resolves to the vendored copy. The torchvision shim is still needed (transformers imports
it on the way to EsmForMaskedLM, and the env's torchvision is ABI-mismatched).

All CLI args pass straight through to antiGen.train.main.
"""
import os
import sys


import transformers.utils.import_utils as _iu  # noqa: E402
_iu._torchvision_available = False
_iu._torchvision_version = "N/A"

import antiGen  # noqa: E402

from antiGen.train import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
