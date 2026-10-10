"""Tests for modules/config.py."""

from __future__ import annotations

import pytest

# ============================================================================
# Constants
# ============================================================================


def test_ut_kind_constant():
    from modules.config import UT_KIND

    assert UT_KIND == "ut"


def test_st_kind_constant():
    from modules.config import ST_KIND

    assert ST_KIND == "st"


def test_case_comparison_sections():
    from modules.config import CASE_COMPARISON_SECTIONS

    assert CASE_COMPARISON_SECTIONS == ("matched", "cpu_gpu_only", "npu_only", "manual_review")


def test_config_path_points_to_workflow_scope_json():
    from modules.config import CONFIG_PATH

    assert CONFIG_PATH.name == "workflow_scope.json"
    assert CONFIG_PATH.parent.name == "config"
    assert "ascend-ci-case-diff-scan" in str(CONFIG_PATH)


# ============================================================================
# Dataclasses
# ============================================================================


class TestWorkflowConfig:
    def test_construction(self):
        from modules.config import WorkflowConfig

        cfg = WorkflowConfig(ignored_workflows=("docker-*.yml", "doc.yml"))
        assert cfg.ignored_workflows == ("docker-*.yml", "doc.yml")

    def test_immutability(self):
        from dataclasses import FrozenInstanceError

        from modules.config import WorkflowConfig

        cfg = WorkflowConfig(ignored_workflows=("docker-*.yml",))
        with pytest.raises(FrozenInstanceError):
            cfg.ignored_workflows = ("new",)  # type: ignore[misc]

    def test_default_empty(self):
        from modules.config import WorkflowConfig

        cfg = WorkflowConfig(ignored_workflows=())
        assert cfg.ignored_workflows == ()


class TestWorkflowInfo:
    def test_construction(self):
        from modules.config import WorkflowInfo

        info = WorkflowInfo(
            workflow_name="GPU Unit Tests",
            workflow_path=".github/workflows/gpu_unit_tests.yml",
            file_name="gpu_unit_tests.yml",
            workflow_kind="gpu",
            pair_key="gpu_unit_tests",
        )
        assert info.workflow_name == "GPU Unit Tests"
        assert info.workflow_path == ".github/workflows/gpu_unit_tests.yml"
        assert info.file_name == "gpu_unit_tests.yml"
        assert info.workflow_kind == "gpu"
        assert info.pair_key == "gpu_unit_tests"

    def test_immutability(self):
        from dataclasses import FrozenInstanceError

        from modules.config import WorkflowInfo

        info = WorkflowInfo(
            workflow_name="test",
            workflow_path=".github/workflows/test.yml",
            file_name="test.yml",
            workflow_kind="cpu",
            pair_key="test",
        )
        with pytest.raises(FrozenInstanceError):
            info.workflow_name = "changed"  # type: ignore[misc]


# ============================================================================
# load_text
# ============================================================================


class TestLoadText:
    def test_utf8(self, tmp_path):
        from modules.config import load_text

        path = tmp_path / "test_utf8.txt"
        path.write_text("hello world", encoding="utf-8")
        assert load_text(path) == "hello world"

    def test_utf8_sig(self, tmp_path):
        from modules.config import load_text

        path = tmp_path / "test_utf8_sig.txt"
        path.write_bytes(b"\xef\xbb\xbfhello world")
        # utf-8-sig is tried first and strips the BOM if present.
        result = load_text(path)
        assert result == "hello world"

    def test_gb18030(self, tmp_path):
        from modules.config import load_text

        path = tmp_path / "test_gb18030.txt"
        path.write_text("中文测试", encoding="gb18030")
        assert load_text(path) == "中文测试"

    def test_latin1(self, tmp_path):
        from modules.config import load_text

        path = tmp_path / "test_latin1.txt"
        path.write_bytes(b"caf\xe9")
        result = load_text(path)
        assert result == "café"

    def test_fallback_latin1(self, tmp_path):
        """Bytes invalid in utf-8/utf-8-sig/gb18030 but valid in latin-1 fall through to latin-1.

        Latin-1 maps every byte 0x00-0xFF to a character, so any arbitrary byte sequence
        will be decoded by this fallback. The final ``errors="ignore"`` fallback is a
        safety net that is unreachable with normal byte sequences.
        """
        from modules.config import load_text

        path = tmp_path / "test_bad.bin"
        # Write raw bytes that are invalid in utf-8 but valid in latin-1
        path.write_bytes(b"\xff\xfe\xfd")
        result = load_text(path)
        assert isinstance(result, str)
        assert len(result) > 0
        # Latin-1 maps \xff → ÿ, \xfe → þ, \xfd → ý
        assert result == "ÿþý"

    def test_empty_file(self, tmp_path):
        from modules.config import load_text

        path = tmp_path / "empty.txt"
        path.write_text("", encoding="utf-8")
        assert load_text(path) == ""


# ============================================================================
# load_config
# ============================================================================


class TestLoadConfig:
    def test_loads_ignored_workflows(self, temp_json_config):
        from modules.config import load_config

        cfg = load_config(temp_json_config)
        assert cfg.ignored_workflows == ("docker-*.yml", "doc.yml")

    def test_empty_ignored_list(self, tmp_path):
        from modules.config import load_config

        config_path = tmp_path / "empty_config.json"
        config_path.write_text('{"ignored_workflows": []}', encoding="utf-8")
        cfg = load_config(config_path)
        assert cfg.ignored_workflows == ()

    def test_missing_ignored_key_defaults_to_empty(self, tmp_path):
        from modules.config import load_config

        config_path = tmp_path / "minimal_config.json"
        config_path.write_text("{}", encoding="utf-8")
        cfg = load_config(config_path)
        assert cfg.ignored_workflows == ()


# ============================================================================
# is_ignored_workflow
# ============================================================================


class TestIsIgnoredWorkflow:
    def test_exact_match(self, sample_config):
        from modules.config import is_ignored_workflow

        assert is_ignored_workflow("doc.yml", sample_config) is True

    def test_glob_match(self, sample_config):
        from modules.config import is_ignored_workflow

        assert is_ignored_workflow("docker-build.yml", sample_config) is True
        assert is_ignored_workflow("docker-push.yml", sample_config) is True

    def test_no_match(self, sample_config):
        from modules.config import is_ignored_workflow

        assert is_ignored_workflow("gpu_unit_tests.yml", sample_config) is False
        assert is_ignored_workflow("npu_unit_tests.yml", sample_config) is False

    def test_case_insensitive(self, sample_config):
        from modules.config import is_ignored_workflow

        assert is_ignored_workflow("DOC.YML", sample_config) is True
        assert is_ignored_workflow("Docker-Build.yml", sample_config) is True

    def test_empty_config_returns_false(self, empty_config):
        from modules.config import is_ignored_workflow

        assert is_ignored_workflow("doc.yml", empty_config) is False
        assert is_ignored_workflow("docker-build.yml", empty_config) is False


# ============================================================================
# validate_repo_root
# ============================================================================


class TestValidateRepoRoot:
    def test_valid_repo(self, temp_workflow_dir):
        from modules.config import validate_repo_root

        # temp_workflow_dir already has .github/workflows/
        validate_repo_root(temp_workflow_dir)

    def test_nonexistent_path(self, tmp_path):
        from modules.config import validate_repo_root

        nonexistent = tmp_path / "nonexistent_dir"
        with pytest.raises(FileNotFoundError, match="does not exist"):
            validate_repo_root(nonexistent)

    def test_path_is_file_not_dir(self, tmp_path):
        from modules.config import validate_repo_root

        a_file = tmp_path / "a_file.txt"
        a_file.write_text("hello")
        with pytest.raises(NotADirectoryError):
            validate_repo_root(a_file)

    def test_no_workflow_directory(self, tmp_path):
        from modules.config import validate_repo_root

        # tmp_path exists but has no .github/workflows/
        with pytest.raises(FileNotFoundError, match=".github/workflows"):
            validate_repo_root(tmp_path)
