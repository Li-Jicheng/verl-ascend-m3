"""Tests for modules/excel.py."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, patch

# ============================================================================
# _xml_text
# ============================================================================


class TestXmlText:
    def test_escapes_ampersand(self):
        from modules.excel import _xml_text

        assert _xml_text("a & b") == "a &amp; b"

    def test_escapes_angle_brackets(self):
        from modules.excel import _xml_text

        result = _xml_text("<test>")
        assert result == "&lt;test&gt;"

    def test_strips_invalid_xml_chars(self):
        from xml.sax.saxutils import escape

        from modules.excel import _xml_text

        result = _xml_text("hello\x00world")
        assert result == escape("helloworld")

    def test_preserves_valid_chars(self):
        from modules.excel import _xml_text

        result = _xml_text("hello world 123")
        assert result == "hello world 123"


# ============================================================================
# _is_valid_xml_char
# ============================================================================


class TestIsValidXmlChar:
    def test_tab_newline_carriage_return(self):
        from modules.excel import _is_valid_xml_char

        assert _is_valid_xml_char("\t") is True
        assert _is_valid_xml_char("\n") is True
        assert _is_valid_xml_char("\r") is True

    def test_printable_ascii(self):
        from modules.excel import _is_valid_xml_char

        assert _is_valid_xml_char("A") is True
        assert _is_valid_xml_char(" ") is True
        assert _is_valid_xml_char("~") is True

    def test_null_is_invalid(self):
        from modules.excel import _is_valid_xml_char

        assert _is_valid_xml_char("\x00") is False

    def test_vertical_tab_is_invalid(self):
        from modules.excel import _is_valid_xml_char

        assert _is_valid_xml_char("\x0b") is False

    def test_chinese_char_is_valid(self):
        from modules.excel import _is_valid_xml_char

        assert _is_valid_xml_char("中") is True


# ============================================================================
# _column_name
# ============================================================================


class TestColumnName:
    def test_single_letter(self):
        from modules.excel import _column_name

        assert _column_name(1) == "A"
        assert _column_name(26) == "Z"

    def test_double_letter(self):
        from modules.excel import _column_name

        assert _column_name(27) == "AA"
        assert _column_name(28) == "AB"
        assert _column_name(52) == "AZ"

    def test_triple_letter(self):
        from modules.excel import _column_name

        assert _column_name(703) == "AAA"

    def test_zero_returns_empty(self):
        from modules.excel import _column_name

        assert _column_name(0) == ""


# ============================================================================
# _column_width
# ============================================================================


class TestColumnWidth:
    def test_within_range(self):
        from modules.excel import _column_width

        widths = (10, 20, 30)
        assert _column_width(1, widths) == 10
        assert _column_width(2, widths) == 20
        assert _column_width(3, widths) == 30

    def test_outside_range_uses_default(self):
        from modules.excel import DEFAULT_COLUMN_WIDTH, _column_width

        widths = (10,)
        assert _column_width(5, widths) == DEFAULT_COLUMN_WIDTH


# ============================================================================
# _excel_multiline
# ============================================================================


class TestExcelMultiline:
    def test_replaces_br(self):
        from modules.excel import _excel_multiline

        assert _excel_multiline("a<br>b") == "a\nb"

    def test_no_br(self):
        from modules.excel import _excel_multiline

        assert _excel_multiline("plain") == "plain"

    def test_multiple_br(self):
        from modules.excel import _excel_multiline

        assert _excel_multiline("a<br>b<br>c") == "a\nb\nc"

    def test_empty_string(self):
        from modules.excel import _excel_multiline

        assert _excel_multiline("") == ""


# ============================================================================
# _ref_sort_key
# ============================================================================


class TestRefSortKey:
    def test_produces_tuple(self):
        from modules.excel import _ref_sort_key

        ref = {"name": "test", "workflow_path": "a.yml", "line_number": 1}
        result = _ref_sort_key(ref)
        assert isinstance(result, tuple)
        assert result[0] == "test"
        assert result[1] == "a.yml"
        assert result[2] == 1


# ============================================================================
# _side_cells
# ============================================================================


class TestSideCells:
    def test_with_full_ref(self):
        from modules.excel import _side_cells

        ref = {
            "workflow_path": "a.yml",
            "line_number": 10,
            "workflow_name": "GPU Tests",
            "job_name": "unit-tests",
            "step_name": "Run pytest",
            "signature": "pytest",
            "raw_command": "pytest tests/",
        }
        cells = _side_cells(ref)
        assert len(cells) == 5
        assert cells[0] == "a.yml"
        assert cells[1] == 10
        assert "GPU Tests" in cells[2]
        assert "unit-tests" in cells[2]
        assert "Run pytest" in cells[2]
        assert cells[3] == "pytest"
        assert cells[4] == "pytest tests/"

    def test_with_none(self):
        from modules.excel import _side_cells

        cells = _side_cells(None)
        assert cells == ["", "", "", "", ""]


# ============================================================================
# Row generation functions (pure)
# ============================================================================


class TestRowGeneration:
    def test_ignored_workflow_rows(self):
        from modules.excel import _ignored_workflow_rows

        report = {"ignored_workflows": ["docker-build.yml", "doc.yml"]}
        rows = _ignored_workflow_rows(report)
        assert rows[0] == ["Workflow Name"]
        assert rows[1] == ["docker-build.yml"]
        assert rows[2] == ["doc.yml"]

    def test_ignored_workflow_rows_empty(self):
        from modules.excel import _ignored_workflow_rows

        report = {"ignored_workflows": []}
        rows = _ignored_workflow_rows(report)
        assert rows == [["Workflow Name"]]

    def test_scanned_workflow_rows(self):
        from modules.excel import _scanned_workflow_rows

        report = {
            "scanned_workflows": [
                {
                    "workflow_name": "wf1",
                    "cpu_gpu_case_count": 10,
                    "npu_supported_case_count": 8,
                }
            ]
        }
        rows = _scanned_workflow_rows(report)
        assert rows[0] == ["Workflow Name", "CPU/GPU Case Count", "NPU Supported Case Count"]
        assert rows[1][1] == 10
        assert rows[1][2] == 8

    def test_case_rows_structure(self):
        from modules.excel import _case_rows

        details = {
            "matched": [],
            "cpu_gpu_only": [],
            "npu_only": [],
            "manual_review": [],
        }
        rows = _case_rows(details, "UT Case Name")
        # Should have at least a header row
        assert len(rows) >= 1
        assert rows[0][0] == "UT Case Name"
        assert "Match Status" in rows[0]

    def test_case_rows_with_non_empty_section(self):
        """Cover L101: _case_rows calls _case_item_rows for non-empty sections."""
        from modules.excel import _case_rows

        details = {
            "matched": [
                {
                    "name": "tests/test_a.py::test_1",
                    "cpu_gpu_refs": [
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
                    ],
                    "npu_refs": [],
                }
            ],
            "cpu_gpu_only": [],
            "npu_only": [],
            "manual_review": [],
        }
        rows = _case_rows(details, "ST Case Name")
        # Header + 1 data row
        assert len(rows) == 2
        assert rows[0][0] == "ST Case Name"
        assert rows[1][0] == "tests/test_a.py::test_1"
        assert rows[1][1] == "Matched"

    def test_case_item_rows_with_paired_refs(self):
        from modules.excel import _case_item_rows

        item = {
            "name": "tests/test_a.py::test_1",
            "cpu_gpu_refs": [
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
            ],
            "npu_refs": [
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
            ],
        }
        rows = _case_item_rows(item, "Matched")
        assert len(rows) == 1
        assert rows[0][0] == "tests/test_a.py::test_1"
        assert rows[0][1] == "Matched"
        # CPU/GPU side
        assert rows[0][2] == "a.yml"
        # NPU side
        assert rows[0][7] == "a_ascend.yml"

    def test_case_item_rows_no_npu(self):
        from modules.excel import _case_item_rows

        item = {
            "name": "tests/test_b.py::test_2",
            "cpu_gpu_refs": [
                {
                    "name": "tests/test_b.py::test_2",
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
        rows = _case_item_rows(item, "CPU/GPU Only")
        assert len(rows) == 1
        # NPU side should be empty strings
        assert rows[0][7] == ""


# ============================================================================
# XML generation functions
# ============================================================================


class TestXmlGeneration:
    def test_content_types_xml_valid(self):
        from modules.excel import _content_types_xml

        xml_str = _content_types_xml(4)
        # Should be valid XML
        root = ET.fromstring(xml_str)
        assert root.tag == "{http://schemas.openxmlformats.org/package/2006/content-types}Types"

    def test_root_rels_xml_valid(self):
        from modules.excel import _root_rels_xml

        xml_str = _root_rels_xml()
        root = ET.fromstring(xml_str)
        assert root.tag == "{http://schemas.openxmlformats.org/package/2006/relationships}Relationships"

    def test_workbook_xml_valid(self):
        from modules.excel import _workbook_xml

        sheets = [
            ("Sheet1", [["A", "B"]], (10, 20)),
            ("Sheet2", [["X"]], (30,)),
        ]
        xml_str = _workbook_xml(sheets)
        root = ET.fromstring(xml_str)
        ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
        sheet_elems = root.findall(f"{ns}sheets/{ns}sheet")
        assert len(sheet_elems) == 2

    def test_styles_xml_valid(self):
        from modules.excel import _styles_xml

        xml_str = _styles_xml()
        root = ET.fromstring(xml_str)
        assert root.tag == "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}styleSheet"

    def test_worksheet_xml_valid(self):
        from modules.excel import _worksheet_xml

        rows = [
            ["Header A", "Header B"],
            ["data1", "data2"],
            [42, 99],
        ]
        xml_str = _worksheet_xml(rows, (20, 30))
        root = ET.fromstring(xml_str)
        ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
        row_elems = root.findall(f"{ns}sheetData/{ns}row")
        assert len(row_elems) == 3

    def test_row_xml_includes_cells(self):
        from modules.excel import _row_xml

        xml_str = _row_xml(1, ["hello", 42, None])
        assert '<row r="1">' in xml_str
        assert "hello" in xml_str
        assert "42" in xml_str

    def test_cell_xml_header_style(self):
        from modules.excel import _cell_xml

        xml_str = _cell_xml(1, 1, "Header")
        assert 's="2"' in xml_str

    def test_cell_xml_data_style(self):
        from modules.excel import _cell_xml

        xml_str = _cell_xml(2, 1, "data")
        assert 's="1"' in xml_str

    def test_cell_xml_integer(self):
        from modules.excel import _cell_xml

        xml_str = _cell_xml(2, 1, 42)
        assert "<v>42</v>" in xml_str
        assert 't="inlineStr"' not in xml_str

    def test_cell_xml_inline_string(self):
        from modules.excel import _cell_xml

        xml_str = _cell_xml(2, 1, "hello")
        assert 't="inlineStr"' in xml_str

    def test_cell_xml_none_value(self):
        from modules.excel import _cell_xml

        xml_str = _cell_xml(2, 1, None)
        # None value → empty inline string
        assert 't="inlineStr"' in xml_str
        assert "<t></t>" in xml_str

    def test_workbook_rels_xml(self):
        from modules.excel import _workbook_rels_xml

        xml_str = _workbook_rels_xml(4)
        root = ET.fromstring(xml_str)
        ns = "{http://schemas.openxmlformats.org/package/2006/relationships}"
        rels = root.findall(f"{ns}Relationship")
        # 4 sheet relationships + 1 styles relationship = 5
        assert len(rels) == 5


# ============================================================================
# write_excel_report (I/O with mocked ZipFile)
# ============================================================================


class TestWriteExcelReport:
    def test_writes_zip_with_four_sheets(self, tmp_path):
        from modules.excel import write_excel_report

        report = {
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

        path = tmp_path / "test_report.xlsx"

        with patch("modules.excel.ZipFile") as mock_zip:
            mock_zip_instance = MagicMock()
            mock_zip.return_value.__enter__.return_value = mock_zip_instance

            write_excel_report(path, report)

            # Verify ZipFile was created with the right path and write mode
            from zipfile import ZIP_DEFLATED

            mock_zip.assert_called_once_with(path, "w", ZIP_DEFLATED)

            # Verify writestr was called for each required file
            written_names = {args[0][0] for args in mock_zip_instance.writestr.call_args_list}
            assert "[Content_Types].xml" in written_names
            assert "_rels/.rels" in written_names
            assert "xl/workbook.xml" in written_names
            assert "xl/_rels/workbook.xml.rels" in written_names
            assert "xl/styles.xml" in written_names
            assert "xl/worksheets/sheet1.xml" in written_names
            assert "xl/worksheets/sheet2.xml" in written_names
            assert "xl/worksheets/sheet3.xml" in written_names
            assert "xl/worksheets/sheet4.xml" in written_names


# ============================================================================
# write_past_commit_excel_report (I/O with mocked ZipFile)
# ============================================================================


class TestWritePastCommitExcelReport:
    def test_writes_zip_with_four_sheets(self, tmp_path):
        from modules.excel import write_past_commit_excel_report

        report = {
            "summary": [],
            "workflow_changes": [],
            "case_details": [],
            "commit_details": [],
        }

        path = tmp_path / "test_past_report.xlsx"

        with patch("modules.excel.ZipFile") as mock_zip:
            mock_zip_instance = MagicMock()
            mock_zip.return_value.__enter__.return_value = mock_zip_instance

            write_past_commit_excel_report(path, report)

            from zipfile import ZIP_DEFLATED

            mock_zip.assert_called_once_with(path, "w", ZIP_DEFLATED)

            written_names = {args[0][0] for args in mock_zip_instance.writestr.call_args_list}
            assert "xl/worksheets/sheet1.xml" in written_names  # Summary
            assert "xl/worksheets/sheet2.xml" in written_names  # Changed Workflows
            assert "xl/worksheets/sheet3.xml" in written_names  # Changed Cases
            assert "xl/worksheets/sheet4.xml" in written_names  # Commit Details


# ============================================================================
# _past_*_rows (pure functions)
# ============================================================================


class TestPastRows:
    def test_past_summary_rows(self):
        from modules.excel import _past_summary_rows

        report = {
            "workflow_changes": [
                {
                    "workflow_path": "gpu_unit_tests.yml",
                    "ut_gap_count": 3,
                    "st_gap_count": 1,
                    "commit_hashes": ("abc123", "def456"),
                }
            ]
        }
        rows = _past_summary_rows(report)
        assert len(rows) >= 2  # header + at least 1 data row
        assert rows[1][0] == "gpu_unit_tests.yml"

    def test_past_workflow_rows(self):
        from modules.excel import _past_workflow_rows

        report = {
            "workflow_changes": [
                {
                    "workflow_path": "gpu_unit_tests.yml",
                    "workflow_status": "modified",
                    "case_count_base": 10,
                    "case_count_head": 12,
                    "ut_gap_count": 2,
                    "st_gap_count": 0,
                    "commit_hashes": ("abc123",),
                }
            ]
        }
        rows = _past_workflow_rows(report)
        assert len(rows) >= 2
        assert rows[1][1] == "modified"

    def test_past_case_rows(self):
        from modules.excel import _past_case_rows

        report = {
            "case_details": [
                {
                    "workflow_path": "gpu_unit_tests.yml",
                    "case_name": "tests/test_new.py::test_feature",
                    "case_kind": "ut",
                    "target": "tests/test_new.py",
                    "line_number": 10,
                    "workflow_context": "GPU Tests / unit-tests / Run pytest",
                    "signature": "pytest",
                    "npu_status": "missing_in_npu_workflows",
                    "npu_refs": [],
                    "commit_hashes": ("abc123",),
                }
            ]
        }
        rows = _past_case_rows(report)
        assert len(rows) >= 2
        assert rows[1][0] == "gpu_unit_tests.yml"
        assert rows[1][1] == "tests/test_new.py::test_feature"
        assert rows[1][2] == "ut"

    def test_past_detail_rows(self):
        from modules.excel import _past_detail_rows

        report = {
            "commit_details": [
                {
                    "commit_hash": "abc123def456",
                    "commit_time": "2026-05-30T00:00:00+00:00",
                    "commit_title": "Add new test",
                    "affected_workflows": ("gpu_unit_tests.yml",),
                }
            ]
        }
        rows = _past_detail_rows(report)
        assert len(rows) >= 2
        assert rows[1][0] == "abc123def456"
        assert rows[1][1] == "2026-05-30T00:00:00+00:00"
        assert rows[1][2] == "Add new test"
