"""Tests for modules/compare.py."""

from __future__ import annotations

# ============================================================================
# Helpers to build case dicts
# ============================================================================


def _make_case(**overrides):
    """Build a minimal case dict with defaults."""
    from modules.workflows import build_case_id, build_display_name

    defaults = {
        "workflow_name": "Test Workflow",
        "workflow_path": ".github/workflows/test.yml",
        "file_name": "test.yml",
        "workflow_kind": "gpu",
        "pair_key": "test",
        "job_name": "test-job",
        "step_name": "test-step",
        "line_number": 10,
        "command_type": "pytest",
        "case_kind": "ut",
        "target": "tests/test_foo.py::test_add",
        "raw_command": "pytest tests/",
        "signature": "pytest pytest-dir",
        "case_id": "ut|pytest|tests/test_foo.py::test_add|pytest|test",
        "display_name": "tests/test_foo.py::test_add",
    }
    defaults.update(overrides)
    # Regenerate case_id and display_name when not overridden,
    # so they stay consistent with any overridden target/signature fields.
    if "case_id" not in overrides:
        defaults["case_id"] = build_case_id(defaults)
    if "display_name" not in overrides:
        defaults["display_name"] = build_display_name(defaults)
    return defaults


# ============================================================================
# make_ref
# ============================================================================


class TestMakeRef:
    def test_creates_ref_with_expected_keys(self):
        from modules.compare import make_ref

        case = _make_case()
        ref = make_ref(case)

        assert set(ref.keys()) == {
            "name",
            "workflow_name",
            "workflow_path",
            "job_name",
            "step_name",
            "line_number",
            "signature",
            "raw_command",
        }

    def test_maps_fields_correctly(self):
        from modules.compare import make_ref

        case = _make_case()
        ref = make_ref(case)

        field_map = {
            "name": "display_name",
            "workflow_name": "workflow_name",
            "workflow_path": "workflow_path",
            "job_name": "job_name",
            "step_name": "step_name",
            "line_number": "line_number",
            "signature": "signature",
            "raw_command": "raw_command",
        }
        for ref_key, case_key in field_map.items():
            assert ref[ref_key] == case[case_key], f"Mismatch for {ref_key}"


# ============================================================================
# summarize_scanned_workflows
# ============================================================================


class TestSummarizeScannedWorkflows:
    def test_empty_groups(self):
        from modules.compare import summarize_scanned_workflows

        result = summarize_scanned_workflows({}, [])
        assert result == []

    def test_single_pair_with_cases(self):
        from modules.compare import summarize_scanned_workflows
        from modules.config import WorkflowInfo

        cpu_info = WorkflowInfo(
            workflow_name="GPU Tests",
            workflow_path=".github/workflows/gpu_unit_tests.yml",
            file_name="gpu_unit_tests.yml",
            workflow_kind="gpu",
            pair_key="gpu_unit_tests",
        )
        npu_info = WorkflowInfo(
            workflow_name="NPU Tests",
            workflow_path=".github/workflows/npu_unit_tests.yml",
            file_name="npu_unit_tests.yml",
            workflow_kind="npu",
            pair_key="gpu_unit_tests",
        )
        grouped = {
            "gpu_unit_tests": {"cpu_gpu": [cpu_info], "npu": [npu_info]},
        }
        cases = [
            _make_case(
                workflow_path=".github/workflows/gpu_unit_tests.yml",
                case_id="gpu:case:1",
            ),
            _make_case(
                workflow_path=".github/workflows/gpu_unit_tests.yml",
                case_id="gpu:case:2",
            ),
            _make_case(
                workflow_path=".github/workflows/npu_unit_tests.yml",
                workflow_kind="npu",
                case_id="npu:case:1",
                display_name="tests/test_foo.py::test_add",
            ),
        ]

        result = summarize_scanned_workflows(grouped, cases)

        assert len(result) == 1
        assert result[0]["cpu_gpu_case_count"] == 2
        assert result[0]["npu_supported_case_count"] == 1
        assert ".github/workflows/gpu_unit_tests.yml" in result[0]["workflow_name"]
        assert ".github/workflows/npu_unit_tests.yml" in result[0]["workflow_name"]

    def test_html_br_formatting(self):
        from modules.compare import summarize_scanned_workflows
        from modules.config import WorkflowInfo

        cpu_info = WorkflowInfo(
            workflow_name="A",
            workflow_path="a.yml",
            file_name="a.yml",
            workflow_kind="gpu",
            pair_key="a",
        )
        npu_info = WorkflowInfo(
            workflow_name="B",
            workflow_path="b.yml",
            file_name="b.yml",
            workflow_kind="npu",
            pair_key="a",
        )
        grouped = {"a": {"cpu_gpu": [cpu_info], "npu": [npu_info]}}

        result = summarize_scanned_workflows(grouped, [])
        assert "<br>" in result[0]["workflow_name"]


# ============================================================================
# compare_group_cases
# ============================================================================


class TestCompareGroupCases:
    def test_empty_both_sides(self):
        from modules.compare import compare_group_cases

        result = compare_group_cases([], [])

        for section in ("matched", "cpu_gpu_only", "npu_only", "manual_review"):
            assert result[section] == []

    def test_exact_match(self):
        from modules.compare import compare_group_cases

        cpu_case = _make_case(
            command_type="pytest",
            target="tests/test_foo.py::test_add",
            signature="pytest",
        )
        npu_case = _make_case(
            workflow_kind="npu",
            command_type="pytest",
            target="tests/test_foo.py::test_add",
            signature="pytest",
        )

        result = compare_group_cases([cpu_case], [npu_case])

        assert len(result["matched"]) == 1
        assert result["matched"][0]["name"] == "tests/test_foo.py::test_add"
        assert len(result["matched"][0]["cpu_gpu_refs"]) == 1
        assert len(result["matched"][0]["npu_refs"]) == 1
        assert result["cpu_gpu_only"] == []
        assert result["npu_only"] == []
        assert result["manual_review"] == []

    def test_cpu_gpu_only(self):
        from modules.compare import compare_group_cases

        cpu_case = _make_case(
            command_type="pytest",
            target="tests/test_new.py::test_new_feature",
            signature="pytest",
        )

        result = compare_group_cases([cpu_case], [])

        assert result["matched"] == []
        assert len(result["cpu_gpu_only"]) == 1
        assert result["cpu_gpu_only"][0]["name"] == "tests/test_new.py::test_new_feature"
        assert len(result["cpu_gpu_only"][0]["cpu_gpu_refs"]) == 1
        assert result["cpu_gpu_only"][0]["npu_refs"] == []
        assert result["npu_only"] == []
        assert result["manual_review"] == []

    def test_npu_only(self):
        from modules.compare import compare_group_cases

        npu_case = _make_case(
            workflow_kind="npu",
            command_type="pytest",
            target="tests/test_ascend.py::test_ascend_feature",
            signature="pytest",
        )

        result = compare_group_cases([], [npu_case])

        assert result["matched"] == []
        assert result["cpu_gpu_only"] == []
        assert len(result["npu_only"]) == 1
        assert result["npu_only"][0]["name"] == "tests/test_ascend.py::test_ascend_feature"
        assert result["npu_only"][0]["cpu_gpu_refs"] == []
        assert len(result["npu_only"][0]["npu_refs"]) == 1
        assert result["manual_review"] == []

    def test_manual_review_same_target_different_signature(self):
        from modules.compare import compare_group_cases

        cpu_case = _make_case(
            command_type="pytest",
            target="tests/test_common.py::test_feature",
            signature="pytest",
        )
        npu_case = _make_case(
            workflow_kind="npu",
            command_type="pytest",
            target="tests/test_common.py::test_feature",
            signature="pytest --some-flag",
        )

        result = compare_group_cases([cpu_case], [npu_case])

        assert result["matched"] == []
        assert result["cpu_gpu_only"] == []
        assert result["npu_only"] == []
        assert len(result["manual_review"]) == 1
        assert result["manual_review"][0]["name"] == "tests/test_common.py::test_feature"

    def test_mixed_scenario(self):
        """Test a realistic scenario with multiple categories."""
        from modules.compare import compare_group_cases

        cpu_cases = [
            _make_case(target="tests/test_a.py::test_1", signature="pytest"),
            _make_case(target="tests/test_b.py::test_2", signature="pytest"),
            _make_case(target="tests/test_c.py::test_3", signature="pytest"),
        ]
        npu_cases = [
            _make_case(workflow_kind="npu", target="tests/test_a.py::test_1", signature="pytest"),
            _make_case(workflow_kind="npu", target="tests/test_c.py::test_3", signature="pytest --env"),
            _make_case(workflow_kind="npu", target="tests/test_d.py::test_4", signature="pytest"),
        ]

        result = compare_group_cases(cpu_cases, npu_cases)

        # test_a: matched
        assert len(result["matched"]) == 1
        assert result["matched"][0]["name"] == "tests/test_a.py::test_1"

        # test_b: cpu_gpu_only
        assert len(result["cpu_gpu_only"]) == 1
        assert result["cpu_gpu_only"][0]["name"] == "tests/test_b.py::test_2"

        # test_d: npu_only
        assert len(result["npu_only"]) == 1
        assert result["npu_only"][0]["name"] == "tests/test_d.py::test_4"

        # test_c: manual_review (same target, different signatures)
        assert len(result["manual_review"]) == 1
        assert result["manual_review"][0]["name"] == "tests/test_c.py::test_3"

    def test_results_are_sorted(self):
        from modules.compare import compare_group_cases

        cpu_cases = [
            _make_case(target="tests/test_z.py::test_last", signature="pytest"),
            _make_case(target="tests/test_a.py::test_first", signature="pytest"),
        ]

        result = compare_group_cases(cpu_cases, [])
        names = [item["name"] for item in result["cpu_gpu_only"]]
        assert names == sorted(names)

    def test_all_four_sections_present(self):
        from modules.compare import compare_group_cases

        result = compare_group_cases([], [])
        assert set(result.keys()) == {"matched", "cpu_gpu_only", "npu_only", "manual_review"}


# ============================================================================
# compare_cases_by_pair
# ============================================================================


class TestCompareCasesByPair:
    def test_filters_by_case_kind(self):
        from modules.compare import compare_cases_by_pair

        ut_case = _make_case(
            case_kind="ut",
            pair_key="test",
            target="tests/test_a.py::test_1",
            signature="pytest",
        )
        st_case = _make_case(
            case_kind="st",
            pair_key="test",
            command_type="torchrun",
            target="tests/test_trainer.py",
            signature="torchrun",
        )

        ut_result = compare_cases_by_pair([ut_case, st_case], "ut")
        st_result = compare_cases_by_pair([ut_case, st_case], "st")

        # UT result should only see the UT case (as cpu_gpu_only since no NPU)
        assert len(ut_result["cpu_gpu_only"]) == 1
        assert ut_result["cpu_gpu_only"][0]["command_type"] == "pytest"

        # ST result should only see the ST case
        assert len(st_result["cpu_gpu_only"]) == 1
        assert st_result["cpu_gpu_only"][0]["command_type"] == "torchrun"

    def test_aggregates_across_pairs(self):
        from modules.compare import compare_cases_by_pair

        case_a = _make_case(
            pair_key="pair_a",
            target="tests/test_a.py::test_1",
            signature="pytest",
        )
        case_b = _make_case(
            pair_key="pair_b",
            target="tests/test_b.py::test_2",
            signature="pytest",
        )

        result = compare_cases_by_pair([case_a, case_b], "ut")

        assert len(result["cpu_gpu_only"]) == 2
        names = [item["name"] for item in result["cpu_gpu_only"]]
        assert "tests/test_a.py::test_1" in names
        assert "tests/test_b.py::test_2" in names
