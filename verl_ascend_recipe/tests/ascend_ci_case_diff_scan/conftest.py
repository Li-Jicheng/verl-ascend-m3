"""Shared test fixtures and path setup for ascend-ci-case-diff-scan tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add the skill's scripts/ directory to sys.path so that
# "import modules.xxx" and relative imports inside modules work correctly.
_SKILL_SCRIPTS = Path(__file__).resolve().parents[2] / ".agent" / "skills" / "ascend-ci-case-diff-scan" / "scripts"
if not any(Path(p).resolve() == _SKILL_SCRIPTS.resolve() for p in sys.path):
    sys.path.insert(0, str(_SKILL_SCRIPTS))

# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_config():
    """A WorkflowConfig with typical ignored patterns."""
    from modules.config import WorkflowConfig

    return WorkflowConfig(ignored_workflows=("docker-*.yml", "doc.yml", "pre-commit.yml"))


@pytest.fixture
def empty_config():
    """A WorkflowConfig with no ignored patterns."""
    from modules.config import WorkflowConfig

    return WorkflowConfig(ignored_workflows=())


# ---------------------------------------------------------------------------
# Workflow YAML content fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_workflow_yaml():
    """Minimal valid workflow YAML with a pytest step."""
    return """name: GPU Unit Tests
on: [push]
jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - name: Run pytest
        run: pytest tests/
"""


@pytest.fixture
def sample_ascend_workflow_yaml():
    """Minimal NPU workflow YAML with ascend runner."""
    return """name: NPU Unit Tests
on: [push]
jobs:
  unit-tests:
    runs-on: aarch64
    steps:
      - name: Run pytest on NPU
        run: pytest tests/
"""


@pytest.fixture
def sample_torchrun_workflow_yaml():
    """Workflow YAML with a torchrun step."""
    return """name: GPU E2E Tests
on: [push]
jobs:
  e2e-tests:
    runs-on: ubuntu-latest
    steps:
      - name: Run torchrun test
        run: torchrun tests/test_trainer.py
"""


@pytest.fixture
def sample_bash_workflow_yaml():
    """Workflow YAML with a bash test script step."""
    return """name: GPU Bash Tests
on: [push]
jobs:
  bash-tests:
    runs-on: ubuntu-latest
    steps:
      - name: Run bash test
        run: bash tests/test_integration.sh
"""


@pytest.fixture
def sample_multi_step_workflow_yaml():
    """Workflow YAML with multiple jobs and steps."""
    return """name: Multi Step Tests
on: [push]
jobs:
  first-job:
    runs-on: ubuntu-latest
    steps:
      - name: Step one
        run: pytest tests/a/
      - name: Step two
        run: pytest tests/b/
  second-job:
    runs-on: ubuntu-latest
    steps:
      - name: Step three
        run: bash tests/run.sh
"""


# ---------------------------------------------------------------------------
# Temp directory fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_workflow_dir(tmp_path):
    """Create a tmp_path with .github/workflows/ structure."""
    workflows_dir = tmp_path / ".github" / "workflows"
    workflows_dir.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def temp_json_config(tmp_path):
    """Write a temporary workflow_scope.json and return its path."""
    config_path = tmp_path / "workflow_scope.json"
    config_path.write_text(
        json.dumps({"ignored_workflows": ["docker-*.yml", "doc.yml"]}),
        encoding="utf-8",
    )
    return config_path


# ---------------------------------------------------------------------------
# Python test file fixtures (for extractor tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_test_py_content():
    """Python source with various test functions and classes.

    NOTE: This content is parsed by ast.parse() for test function
    extraction only — it is never executed, so undefined references
    and missing imports are intentional.
    """
    return '''
import pytest

def test_simple():
    assert True

def test_with_fixture(tmp_path):
    assert tmp_path.is_dir()

async def test_async_function():
    result = await some_coroutine()
    assert result is not None

class TestMyFeature:
    def test_method_one(self):
        assert 1 + 1 == 2

    def test_method_two(self):
        assert "hello" == "hello"

    async def test_async_method(self):
        pass

class TestAnotherFeature:
    def test_another(self):
        pass

def helper_function():
    """Not a test."""
    pass

class NotATestClass:
    def test_looks_like_test(self):
        """Not a real test class."""
        pass
'''


@pytest.fixture
def sample_test_py_file(tmp_path, sample_test_py_content):
    """Write a sample test .py file and return its path."""
    test_file = tmp_path / "test_sample.py"
    test_file.write_text(sample_test_py_content, encoding="utf-8")
    return test_file


# ---------------------------------------------------------------------------
# Case dict fixtures (for compare/render tests)
# ---------------------------------------------------------------------------

_BASE_UT_CASE = {
    "line_number": 8,
    "command_type": "pytest",
    "case_kind": "ut",
    "target": "tests/test_foo.py::test_add",
    "raw_command": "pytest tests/",
    "signature": "pytest pytest-dir",
}


@pytest.fixture
def sample_cpu_gpu_case():
    """A representative CPU/GPU case dict."""
    return {
        **_BASE_UT_CASE,
        "workflow_name": "GPU Unit Tests",
        "workflow_path": ".github/workflows/gpu_unit_tests.yml",
        "file_name": "gpu_unit_tests.yml",
        "workflow_kind": "gpu",
        "pair_key": "gpu_unit_tests",
        "job_name": "unit-tests",
        "step_name": "Run pytest",
    }


@pytest.fixture
def sample_npu_case():
    """A representative NPU case dict matching the CPU/GPU case."""
    return {
        **_BASE_UT_CASE,
        "workflow_name": "NPU Unit Tests",
        "workflow_path": ".github/workflows/npu_unit_tests.yml",
        "file_name": "npu_unit_tests.yml",
        "workflow_kind": "npu",
        "pair_key": "gpu_unit_tests",
        "job_name": "unit-tests",
        "step_name": "Run pytest on NPU",
    }


@pytest.fixture
def sample_st_case():
    """A representative ST (torchrun) case dict."""
    return {
        "workflow_name": "GPU E2E Tests",
        "workflow_path": ".github/workflows/gpu_e2e.yml",
        "file_name": "gpu_e2e.yml",
        "workflow_kind": "gpu",
        "pair_key": "gpu_e2e",
        "job_name": "e2e-tests",
        "step_name": "Run torchrun",
        "line_number": 8,
        "command_type": "torchrun",
        "case_kind": "st",
        "target": "tests/test_trainer.py",
        "raw_command": "torchrun tests/test_trainer.py",
        "signature": "torchrun tests/test_trainer.py",
    }
