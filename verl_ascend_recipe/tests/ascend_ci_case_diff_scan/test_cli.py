"""Tests for the CLI entry point scan_ascend_ci_case_diff.py."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

# ============================================================================
# parse_args
# ============================================================================


class TestParseArgs:
    def test_required_args(self):
        """parse_args with --repo-root and --output-dir."""
        from scan_ascend_ci_case_diff import parse_args

        with patch.object(sys, "argv", ["scan", "--repo-root", "/fake/repo", "--output-dir", "/fake/out"]):
            args = parse_args()
            assert args.repo_root == "/fake/repo"
            assert args.output_dir == "/fake/out"
            assert args.since_days is None

    def test_with_since_days(self):
        """parse_args with --since-days sets the value."""
        from scan_ascend_ci_case_diff import parse_args

        with patch.object(
            sys, "argv", ["scan", "--repo-root", "/fake/repo", "--output-dir", "/fake/out", "--since-days", "7"]
        ):
            args = parse_args()
            assert args.since_days == 7

    def test_default_since_days_is_none(self):
        """parse_args defaults --since-days to None."""
        from scan_ascend_ci_case_diff import parse_args

        with patch.object(sys, "argv", ["scan", "--repo-root", "/r", "--output-dir", "/o"]):
            args = parse_args()
            assert args.since_days is None

    def test_repo_root_required(self):
        """--repo-root is required."""
        from scan_ascend_ci_case_diff import parse_args

        with patch.object(sys, "argv", ["scan", "--output-dir", "/o"]):
            with pytest.raises(SystemExit):
                parse_args()

    def test_output_dir_required(self):
        """--output-dir is required."""
        from scan_ascend_ci_case_diff import parse_args

        with patch.object(sys, "argv", ["scan", "--repo-root", "/r"]):
            with pytest.raises(SystemExit):
                parse_args()


# ============================================================================
# main
# ============================================================================


class TestMain:
    def test_since_days_zero_raises_value_error(self):
        """--since-days 0 raises ValueError."""
        from scan_ascend_ci_case_diff import main

        with patch.object(sys, "argv", ["scan", "--repo-root", "/r", "--output-dir", "/o", "--since-days", "0"]):
            with pytest.raises(ValueError, match="positive integer"):
                main()

    def test_since_days_negative_raises_value_error(self):
        """--since-days -1 raises ValueError."""
        from scan_ascend_ci_case_diff import main

        with patch.object(sys, "argv", ["scan", "--repo-root", "/r", "--output-dir", "/o", "--since-days", "-1"]):
            with pytest.raises(ValueError, match="positive integer"):
                main()

    def test_main_basic_flow(self, tmp_path):
        """main() runs full report generation when valid args are given."""
        from scan_ascend_ci_case_diff import main

        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        wf_dir = repo_root / ".github" / "workflows"
        wf_dir.mkdir(parents=True)
        (wf_dir / "gpu_unit_tests.yml").write_text(
            "name: GPU Tests\n"
            "on: [push]\n"
            "jobs:\n"
            "  test:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: pytest tests/\n",
            encoding="utf-8",
        )
        output_dir = tmp_path / "output"

        with patch.object(
            sys,
            "argv",
            [
                "scan",
                "--repo-root",
                str(repo_root),
                "--output-dir",
                str(output_dir),
            ],
        ):
            exit_code = main()

        assert exit_code == 0
        assert (output_dir / "report.md").is_file()
        assert (output_dir / "report.xlsx").is_file()

    def test_main_with_since_days(self, tmp_path):
        """main() with --since-days also writes past-N reports."""
        from scan_ascend_ci_case_diff import main

        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        wf_dir = repo_root / ".github" / "workflows"
        wf_dir.mkdir(parents=True)
        (wf_dir / "gpu_unit_tests.yml").write_text(
            "name: GPU Tests\n"
            "on: [push]\n"
            "jobs:\n"
            "  test:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: pytest tests/\n",
            encoding="utf-8",
        )
        output_dir = tmp_path / "output"

        with (
            patch.object(
                sys,
                "argv",
                [
                    "scan",
                    "--repo-root",
                    str(repo_root),
                    "--output-dir",
                    str(output_dir),
                    "--since-days",
                    "7",
                ],
            ),
            patch("scan_ascend_ci_case_diff.build_past_commit_report") as mock_past,
        ):
            mock_past.return_value = {
                "repo_root": str(repo_root),
                "generated_at": "2026-06-07T00:00:00+00:00",
                "since_days": 7,
                "commit_count": 0,
                "summary": [],
                "workflow_changes": [],
                "case_details": [],
                "commit_details": [],
            }
            exit_code = main()

        assert exit_code == 0
        assert (output_dir / "report.md").is_file()
        assert (output_dir / "report.xlsx").is_file()
        assert (output_dir / "report-past-7.md").is_file()
        assert (output_dir / "report-past-7.xlsx").is_file()
        mock_past.assert_called_once()

    def test_main_invalid_repo_root_raises(self, tmp_path):
        """main() raises FileNotFoundError for nonexistent repo-root."""
        from scan_ascend_ci_case_diff import main

        nonexistent = tmp_path / "nonexistent"
        output_dir = tmp_path / "output"

        with patch.object(
            sys,
            "argv",
            [
                "scan",
                "--repo-root",
                str(nonexistent),
                "--output-dir",
                str(output_dir),
            ],
        ):
            with pytest.raises(FileNotFoundError):
                main()
