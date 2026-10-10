"""Apply verl source patches required by true_on_policy (idempotent, version-aware)."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from .verl_patch_selector import VerlPatchPlan, select_verl_source_patches

logger = logging.getLogger(__name__)

_ROLLOUT_CONFIG_REL = Path("verl/workers/config/rollout.py")
_LLM_SERVER_REL = Path("verl/workers/rollout/llm_server.py")
_PP_FN_MARKER = "def ensure_rollout_config"
_INVALIDATE_MODULE_PREFIXES = (
    "verl.workers.config.rollout",
    "verl.workers.config",
    "verl.workers.rollout",
)


def _recipe_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def _verl_root(recipe_dir: Path) -> Path:
    return recipe_dir.parent.parent


def _git_apply_check(verl_root: Path, patch_file: Path, *, reverse: bool = False) -> bool:
    cmd = ["git", "-C", str(verl_root), "apply"]
    if reverse:
        cmd.append("-R")
    cmd.extend(["--check", str(patch_file)])
    return subprocess.run(cmd, capture_output=True, check=False).returncode == 0


def _invalidate_patched_verl_modules() -> None:
    """Drop cached rollout modules so imports see git-applied source."""
    for name in list(sys.modules):
        for prefix in _INVALIDATE_MODULE_PREFIXES:
            if name == prefix or name.startswith(f"{prefix}."):
                del sys.modules[name]
                break


def _verify_pp_seed_consistency(verl_root: Path) -> None:
    rollout_config = verl_root / _ROLLOUT_CONFIG_REL
    llm_server = verl_root / _LLM_SERVER_REL
    rollout_text = rollout_config.read_text(encoding="utf-8")
    llm_text = llm_server.read_text(encoding="utf-8")
    has_fn = _PP_FN_MARKER in rollout_text
    has_import = "ensure_rollout_config" in llm_text
    if has_import != has_fn:
        raise RuntimeError(
            "[true_on_policy] inconsistent verl tree: llm_server references "
            "ensure_rollout_config but rollout.py is missing it. "
            f"Restore a clean tree with: git -C {verl_root} checkout -- ."
        )


def _apply_patch_file(verl_root: Path, patch_file: Path) -> None:
    if _git_apply_check(verl_root, patch_file, reverse=True):
        logger.info("[true_on_policy] patch already applied: %s", patch_file.name)
        return

    if not _git_apply_check(verl_root, patch_file):
        raise RuntimeError(
            f"[true_on_policy] patch does not apply cleanly: {patch_file.name} (verl tree at {verl_root})."
        )

    subprocess.run(
        ["git", "-C", str(verl_root), "apply", str(patch_file)],
        check=True,
    )
    logger.warning("[true_on_policy] applied %s", patch_file.name)


def apply_verl_source_patches() -> None:
    """Apply version-selected verl source patches (NPU PP + per-request seed + MindSpeed)."""
    recipe_dir = _recipe_dir()
    verl_root = _verl_root(recipe_dir)
    rollout_config = verl_root / _ROLLOUT_CONFIG_REL

    if not rollout_config.is_file():
        raise FileNotFoundError(
            f"[true_on_policy] verl source tree not found at {verl_root}; "
            "copy this recipe to verl/verl_ascend_recipe/true_on_policy first."
        )

    plan = select_verl_source_patches(recipe_dir, verl_root)
    logger.info(
        "[true_on_policy] detected verl %s (%s)",
        plan.verl_version,
        plan.verl_branch,
    )
    for reason in plan.skip_reasons:
        logger.info("[true_on_policy] %s", reason)

    if not plan.patch_files:
        logger.info("[true_on_policy] no verl source patches needed")
    else:
        for patch_file in plan.patch_files:
            _apply_patch_file(verl_root, patch_file)

    _verify_pp_seed_consistency(verl_root)
    _invalidate_patched_verl_modules()


def get_verl_patch_plan() -> VerlPatchPlan:
    """Return the patch plan for the current verl tree (for diagnostics/tests)."""
    recipe_dir = _recipe_dir()
    verl_root = _verl_root(recipe_dir)
    return select_verl_source_patches(recipe_dir, verl_root)
