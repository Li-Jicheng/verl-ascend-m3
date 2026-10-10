"""Tests for modules/render.py."""

from __future__ import annotations

# ============================================================================
# render_reference_details
# ============================================================================


class TestRenderReferenceDetails:
    def test_with_full_ref(self):
        from modules.render import render_reference_details

        ref = {
            "name": "tests/test_foo.py::test_add",
            "workflow_path": ".github/workflows/test.yml",
            "line_number": 10,
            "workflow_name": "GPU Tests",
            "job_name": "unit-tests",
            "step_name": "Run pytest",
            "signature": "pytest",
            "raw_command": "pytest tests/",
        }
        lines = render_reference_details("  ", "CPU/GPU", ref)
        text = "\n".join(lines)

        assert "CPU/GPU:" in text
        assert ".github/workflows/test.yml" in text
        assert "10" in text
        assert "GPU Tests" in text
        assert "unit-tests" in text
        assert "Run pytest" in text

    def test_with_none_ref(self):
        from modules.render import render_reference_details

        lines = render_reference_details("  ", "NPU", None)
        text = "\n".join(lines)

        assert "NPU:" in text
        assert "None" in text


# ============================================================================
# render_adjacent_pairs
# ============================================================================


class TestRenderAdjacentPairs:
    def test_both_non_empty(self):
        from modules.render import render_adjacent_pairs

        cpu_refs = [
            {
                "name": "tests/test_a.py::test_1",
                "workflow_path": "a.yml",
                "line_number": 1,
                "workflow_name": "A",
                "job_name": "j",
                "step_name": "s",
                "signature": "pytest",
                "raw_command": "pytest tests/",
            }
        ]
        npu_refs = [
            {
                "name": "tests/test_a.py::test_1",
                "workflow_path": "a_ascend.yml",
                "line_number": 2,
                "workflow_name": "A Ascend",
                "job_name": "j",
                "step_name": "s",
                "signature": "pytest",
                "raw_command": "pytest tests/",
            }
        ]

        lines = render_adjacent_pairs(cpu_refs, npu_refs)
        text = "\n".join(lines)

        assert "Pair 1" in text
        assert "CPU/GPU" in text
        assert "NPU" in text

    def test_one_empty(self):
        from modules.render import render_adjacent_pairs

        cpu_refs = [
            {
                "name": "tests/test_a.py::test_1",
                "workflow_path": "a.yml",
                "line_number": 1,
                "workflow_name": "A",
                "job_name": "j",
                "step_name": "s",
                "signature": "pytest",
                "raw_command": "pytest tests/",
            }
        ]

        lines = render_adjacent_pairs(cpu_refs, [])
        text = "\n".join(lines)

        assert "Pair 1" in text
        assert "CPU/GPU" in text
        # NPU ref is None but still rendered
        assert "NPU:" in text

    def test_both_empty(self):
        from modules.render import render_adjacent_pairs

        lines = render_adjacent_pairs([], [])
        text = "\n".join(lines)

        assert "None" in text


# ============================================================================
# render_table
# ============================================================================


class TestRenderTable:
    def test_simple_table(self):
        from modules.render import render_table

        result = render_table(["Col A", "Col B"], [["1", "2"], ["3", "4"]])
        text = "\n".join(result)

        assert len(result) == 4
        assert "Col A" in text
        assert "Col B" in text
        assert "---" in text
        assert "1" in text
        assert "4" in text

    def test_empty_rows(self):
        from modules.render import render_table

        result = render_table(["Header"], [])
        text = "\n".join(result)

        assert "Header" in text
        assert "---" in text
        # Only header + separator, no data rows
        assert len(result) == 2


# ============================================================================
# render_case_section
# ============================================================================


class TestRenderCaseSection:
    def test_empty_items(self):
        from modules.render import render_case_section

        lines = render_case_section("Test Section", [])
        text = "\n".join(lines)

        assert "### Test Section" in text
        assert "- None" in text

    def test_with_items(self):
        from modules.render import render_case_section

        items = [
            {
                "name": "tests/test_foo.py::test_add",
                "command_type": "pytest",
                "signature": "pytest",
                "cpu_gpu_refs": [
                    {
                        "name": "tests/test_foo.py::test_add",
                        "workflow_path": "a.yml",
                        "line_number": 1,
                        "workflow_name": "A",
                        "job_name": "j",
                        "step_name": "s",
                        "signature": "pytest",
                        "raw_command": "pytest tests/",
                    }
                ],
                "npu_refs": [],
            }
        ]

        lines = render_case_section("Matched Cases", items)
        text = "\n".join(lines)

        assert "### Matched Cases" in text
        assert "tests/test_foo.py::test_add" in text
        assert "CPU/GPU" in text


# ============================================================================
# markdown_table_multiline
# ============================================================================


class TestMarkdownTableMultiline:
    def test_replaces_br(self):
        from modules.render import markdown_table_multiline

        result = markdown_table_multiline("a<br>b")
        assert result == "a<br/>b"

    def test_no_br(self):
        from modules.render import markdown_table_multiline

        result = markdown_table_multiline("plain text")
        assert result == "plain text"

    def test_multiple_br(self):
        from modules.render import markdown_table_multiline

        result = markdown_table_multiline("a<br>b<br>c")
        assert result == "a<br/>b<br/>c"

    def test_empty_string(self):
        from modules.render import markdown_table_multiline

        result = markdown_table_multiline("")
        assert result == ""


# ============================================================================
# render_scanned_workflows
# ============================================================================


class TestRenderScannedWorkflows:
    def test_empty(self):
        from modules.render import render_scanned_workflows

        lines = render_scanned_workflows([])
        text = "\n".join(lines)

        assert "No workflows were scanned" in text

    def test_with_rows(self):
        from modules.render import render_scanned_workflows

        rows = [
            {
                "workflow_name": "gpu_unit_tests.yml (GPU Unit Tests)",
                "cpu_gpu_case_count": 42,
                "npu_supported_case_count": 38,
            }
        ]

        lines = render_scanned_workflows(rows)
        text = "\n".join(lines)

        assert "Workflow Name" in text
        assert "CPU/GPU Case Count" in text
        assert "NPU Supported Case Count" in text
        assert "gpu_unit_tests.yml" in text
        assert "42" in text
        assert "38" in text


# ============================================================================
# render_report (integration)
# ============================================================================


class TestRenderReport:
    def test_full_report_structure(self):
        from modules.render import render_report

        report = {
            "repo_root": "/fake/repo",
            "generated_at": "2026-05-30T00:00:00+00:00",
            "ignored_workflows": ["docker-build.yml"],
            "scanned_workflows": [],
            "ut_details": {
                "matched": [],
                "cpu_gpu_only": [],
                "npu_only": [],
                "manual_review": [],
            },
            "st_details": {
                "matched": [],
                "cpu_gpu_only": [],
                "npu_only": [],
                "manual_review": [],
            },
        }

        result = render_report(report)

        assert "# Ascend CI Workflow and Case Alignment Report" in result
        assert "## Ignored Workflows" in result
        assert "docker-build.yml" in result
        assert "## Scanned Workflows" in result
        assert "## UT Case Details" in result
        assert "## ST Case Details" in result
        assert "### Matched Cases" in result
        assert "### CPU/GPU-only Cases" in result
        assert "### NPU-only Cases" in result
        assert "### Manual Review" in result

    def test_empty_ignored_workflows(self):
        from modules.render import render_report

        report = {
            "repo_root": "/fake/repo",
            "generated_at": "2026-05-30T00:00:00+00:00",
            "ignored_workflows": [],
            "scanned_workflows": [],
            "ut_details": {
                "matched": [],
                "cpu_gpu_only": [],
                "npu_only": [],
                "manual_review": [],
            },
            "st_details": {
                "matched": [],
                "cpu_gpu_only": [],
                "npu_only": [],
                "manual_review": [],
            },
        }

        result = render_report(report)
        assert "No workflows were ignored" in result

    def test_with_case_details(self):
        from modules.render import render_report

        report = {
            "repo_root": "/fake/repo",
            "generated_at": "2026-05-30T00:00:00+00:00",
            "ignored_workflows": [],
            "scanned_workflows": [],
            "ut_details": {
                "matched": [
                    {
                        "name": "tests/test_foo.py::test_add",
                        "command_type": "pytest",
                        "signature": "pytest",
                        "cpu_gpu_refs": [
                            {
                                "name": "tests/test_foo.py::test_add",
                                "workflow_path": "a.yml",
                                "line_number": 1,
                                "workflow_name": "A",
                                "job_name": "j",
                                "step_name": "s",
                                "signature": "pytest",
                                "raw_command": "pytest tests/",
                            }
                        ],
                        "npu_refs": [],
                    }
                ],
                "cpu_gpu_only": [],
                "npu_only": [],
                "manual_review": [],
            },
            "st_details": {
                "matched": [],
                "cpu_gpu_only": [],
                "npu_only": [],
                "manual_review": [],
            },
        }

        result = render_report(report)
        assert "tests/test_foo.py::test_add" in result

    def test_result_ends_with_newline(self):
        from modules.render import render_report

        report = {
            "repo_root": "/fake/repo",
            "generated_at": "2026-05-30T00:00:00+00:00",
            "ignored_workflows": [],
            "scanned_workflows": [],
            "ut_details": {
                "matched": [],
                "cpu_gpu_only": [],
                "npu_only": [],
                "manual_review": [],
            },
            "st_details": {
                "matched": [],
                "cpu_gpu_only": [],
                "npu_only": [],
                "manual_review": [],
            },
        }

        result = render_report(report)
        assert result.endswith("\n")


# ============================================================================
# render_past_commit_report
# ============================================================================


class TestRenderPastCommitReport:
    def test_empty_report(self):
        from modules.render import render_past_commit_report

        report = {
            "repo_root": "/fake/repo",
            "since_days": 7,
            "commit_count": 0,
            "summary": [],
            "workflow_changes": [],
            "case_details": [],
            "commit_details": [],
        }

        result = render_past_commit_report(report)

        assert "# Ascend CI Past N Days Commit Analysis" in result
        assert "/fake/repo" in result
        assert "7" in result
        assert "0" in result

    def test_with_data(self):
        from modules.render import render_past_commit_report

        report = {
            "repo_root": "/fake/repo",
            "since_days": 7,
            "commit_count": 2,
            "summary": [
                {
                    "affected_path": "gpu_unit_tests.yml",
                    "ut_gap_count": 3,
                    "st_gap_count": 1,
                }
            ],
            "workflow_changes": [
                {
                    "workflow_path": "gpu_unit_tests.yml",
                    "workflow_status": "modified",
                    "case_count_base": 10,
                    "case_count_head": 12,
                    "ut_gap_count": 2,
                    "st_gap_count": 0,
                    "commit_hashes": ("abc123def456",),
                }
            ],
            "case_details": [
                {
                    "case_name": "tests/test_new.py::test_feature",
                    "case_kind": "ut",
                    "workflow_path": "gpu_unit_tests.yml",
                    "line_number": 10,
                    "workflow_context": "GPU Tests / unit-tests / Run pytest",
                    "signature": "pytest",
                    "raw_command": "pytest tests/",
                    "npu_status": "missing_in_npu_workflows",
                    "npu_refs": [],
                    "commit_hashes": ("abc123def456",),
                }
            ],
            "commit_details": [
                {
                    "commit_hash": "abc123def456",
                    "commit_time": "2026-05-30T00:00:00+00:00",
                    "commit_title": "Add new test",
                    "changed_files": ("tests/test_new.py",),
                    "affected_workflows": ("gpu_unit_tests.yml",),
                }
            ],
        }

        result = render_past_commit_report(report)

        assert "## Summary" in result
        assert "## Changed Workflows" in result
        assert "## Changed Case Details" in result
        assert "## Commit Details" in result
        assert "gpu_unit_tests.yml" in result
        assert "tests/test_new.py::test_feature" in result

    def test_with_npu_refs(self):
        """Cover L225-227: rendering non-empty npu_refs."""
        from modules.render import render_past_commit_report

        report = {
            "repo_root": "/fake/repo",
            "since_days": 7,
            "commit_count": 1,
            "summary": [],
            "workflow_changes": [],
            "case_details": [
                {
                    "case_name": "tests/test_foo.py::test_bar",
                    "case_kind": "ut",
                    "workflow_path": "gpu.yml",
                    "line_number": 10,
                    "workflow_context": "GPU Tests / unit-tests / Run pytest",
                    "signature": "pytest",
                    "raw_command": "pytest tests/",
                    "npu_status": "aligned",
                    "npu_refs": [
                        {
                            "workflow_name": "NPU Tests",
                            "job_name": "unit-tests",
                            "step_name": "Run pytest",
                            "workflow_path": "npu.yml",
                            "line_number": 12,
                        }
                    ],
                    "commit_hashes": ("abc123",),
                }
            ],
            "commit_details": [],
        }

        result = render_past_commit_report(report)

        assert "NPU refs:" in result
        assert "NPU Tests" in result
        assert "npu.yml" in result
        assert "NPU refs: None" not in result
