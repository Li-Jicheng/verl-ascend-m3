"""true_on_policy patch entry — loaded via VERL_USE_EXTERNAL_MODULES on import verl."""

import importlib

from .apply_verl_source_patches import apply_verl_source_patches
from .apply_vllm_ascend_source_patches import apply_vllm_ascend_source_patches

# Source patches must land before importing patched verl / vllm-ascend rollout modules.
# Use importlib (not a top-level import) so npu_true_on_policy_patch does not load
# verl.workers.config.rollout before git apply can update rollout.py.
apply_verl_source_patches()
apply_vllm_ascend_source_patches()

_runtime = importlib.import_module(".npu_true_on_policy_patch", __package__)
_runtime.apply_batch_consistency_patches()
