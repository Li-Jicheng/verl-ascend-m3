"""Select verl source patches based on detected verl version and upstream state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    from packaging.version import parse as parse_version
except ImportError:  # pragma: no cover - packaging ships with verl
    parse_version = None

_PATCH_DIR_NAME = "verl_patches"
_PP_MARKER = "ensure_rollout_config"
_MINDSPEED_MARKER = "use_flash_attn_npu_batch_invariant"
_PER_REQUEST_SEED_MARKER = "maybe_apply_per_request_seed"

COMBINED_PP_SEED_PATCH_V0_8_0 = "verl_npu_pp_per_request_seed_v0.8.0.patch"
COMBINED_PP_SEED_PATCH_MAIN = "verl_npu_pp_per_request_seed_main.patch"
MINDSPEED_PATCH = "verl_mindspeed_batch_invariant.patch"
PER_REQUEST_SEED_PATCH_V0_8_0 = "verl_per_request_seed_v0.8.0.patch"
PER_REQUEST_SEED_PATCH_MAIN = "verl_per_request_seed_main.patch"


@dataclass(frozen=True, slots=True)
class VerlPatchPlan:
    verl_version: str
    verl_branch: str
    patch_files: tuple[Path, ...]
    skip_reasons: tuple[str, ...]


def _read_verl_version(verl_root: Path) -> str:
    version_file = verl_root / "verl" / "version" / "version"
    if version_file.is_file():
        return version_file.read_text(encoding="utf-8").strip()
    return "unknown"


def _is_verl_0_8_x(version: str) -> bool:
    if parse_version is not None:
        try:
            parsed = parse_version(version)
            return parsed.major == 0 and parsed.minor == 8
        except Exception:
            pass
    return version.startswith("0.8")


def _verl_branch_label(version: str) -> str:
    return "release/v0.8.x" if _is_verl_0_8_x(version) else "main"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _has_npu_pp_support(verl_root: Path) -> bool:
    rollout_config = verl_root / "verl" / "workers" / "config" / "rollout.py"
    return rollout_config.is_file() and f"def {_PP_MARKER}" in _read_text(rollout_config)


def _has_mindspeed_batch_invariant_fix(verl_root: Path) -> bool:
    transformer_impl = verl_root / "verl" / "workers" / "engine" / "mindspeed" / "transformer_impl.py"
    return transformer_impl.is_file() and _MINDSPEED_MARKER in _read_text(transformer_impl)


def _has_per_request_seed_support(verl_root: Path) -> bool:
    rollout_utils = verl_root / "verl" / "workers" / "rollout" / "utils.py"
    return rollout_utils.is_file() and _PER_REQUEST_SEED_MARKER in _read_text(rollout_utils)


def _combined_pp_seed_patch_name(version: str) -> str:
    return COMBINED_PP_SEED_PATCH_V0_8_0 if _is_verl_0_8_x(version) else COMBINED_PP_SEED_PATCH_MAIN


def _per_request_seed_only_patch_name(version: str) -> str:
    return PER_REQUEST_SEED_PATCH_V0_8_0 if _is_verl_0_8_x(version) else PER_REQUEST_SEED_PATCH_MAIN


def select_verl_source_patches(recipe_dir: Path, verl_root: Path) -> VerlPatchPlan:
    """Return ordered patch files to apply; empty when upstream already contains all changes."""
    patch_dir = recipe_dir / "patch" / _PATCH_DIR_NAME
    version = _read_verl_version(verl_root)
    branch = _verl_branch_label(version)

    patch_files: list[Path] = []
    skip_reasons: list[str] = []

    has_pp = _has_npu_pp_support(verl_root)
    has_seed = _has_per_request_seed_support(verl_root)

    if has_pp and has_seed:
        skip_reasons.append(
            f"skip PP+per-request seed patch: upstream already contains {_PP_MARKER} and {_PER_REQUEST_SEED_MARKER}"
        )
    elif has_pp and not has_seed:
        skip_reasons.append(f"skip PP portion: upstream already contains {_PP_MARKER}")
        patch_files.append(patch_dir / _per_request_seed_only_patch_name(version))
    else:
        patch_files.append(patch_dir / _combined_pp_seed_patch_name(version))

    if _has_mindspeed_batch_invariant_fix(verl_root):
        skip_reasons.append(f"skip MindSpeed patch: upstream already contains {_MINDSPEED_MARKER}")
    else:
        patch_files.append(patch_dir / MINDSPEED_PATCH)

    for patch_file in patch_files:
        if not patch_file.is_file():
            raise FileNotFoundError(f"[true_on_policy] patch file not found: {patch_file}")

    return VerlPatchPlan(
        verl_version=version,
        verl_branch=branch,
        patch_files=tuple(patch_files),
        skip_reasons=tuple(skip_reasons),
    )
