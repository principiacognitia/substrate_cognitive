from vte.core.schema import REQUIRED_TRACE_COLUMNS, validate_trace_columns


def test_required_trace_schema_passes():
    result = validate_trace_columns(REQUIRED_TRACE_COLUMNS)

    assert result.ok
    assert result.missing_columns == ()


def test_required_trace_schema_reports_missing_columns():
    columns = [col for col in REQUIRED_TRACE_COLUMNS if col != "heading"]

    result = validate_trace_columns(columns)

    assert not result.ok
    assert result.missing_columns == ("heading",)


def test_trace_schema_allows_extra_columns():
    columns = list(REQUIRED_TRACE_COLUMNS) + ["unrecognized_lab_column"]

    result = validate_trace_columns(columns)

    assert result.ok
    assert result.extra_columns == ("unrecognized_lab_column",)