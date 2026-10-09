"""Keep notebook examples executable against the current native report API."""

import ast
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest

from pimqc.reporting import (
    NarrativeStatsReporter,
    ReportInput,
    VisualAssetReporter,
)


def _cells():
    path = Path(__file__).parents[2] / "examples" / "interactive_tutorial.ipynb"
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def _python_source(cell):
    """Return code-cell source with IPython line magics omitted for AST use."""
    return "".join(
        line
        for line in cell["source"]
        if not line.lstrip().startswith("%")
    )


def test_tutorial_code_cells_compile():
    """Parse every code cell without executing expensive processing stages."""
    for index, cell in enumerate(_cells()):
        if cell["cell_type"] == "code":
            ast.parse(
                _python_source(cell), filename=f"tutorial cell {index}"
            )


def test_tutorial_does_not_embed_contract_assertions():
    """Keep API invariants out of executable tutorial cells."""
    for index, cell in enumerate(_cells()):
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse(
            _python_source(cell), filename=f"tutorial cell {index}"
        )
        assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree))


@pytest.mark.parametrize(
    "correction_names",
    [
        ["Intra-batch corrected", "Inter-batch corrected"],
        ["SERRF"],
    ],
)
def test_tutorial_report_cell_uses_current_api(correction_names, tmp_path):
    """Execute the actual report cell with signature-checked report writers."""
    visual = create_autospec(VisualAssetReporter)
    narrative = create_autospec(NarrativeStatsReporter)
    assets = {"02_PCA_Scatter_Dashboard": "assets/02_PCA_Scatter_Dashboard.svg"}
    visual.return_value.compile_assessor_report.return_value = assets
    namespace = {
        "VisualAssetReporter": visual,
        "NarrativeStatsReporter": narrative,
        "ReportInput": ReportInput,
        "OUTPUT_DIR": tmp_path,
        "EXPORT_PDF": False,
        "datetime": datetime,
        "pimqc": SimpleNamespace(__version__="test-version"),
        "params": {},
        "raw_dataset": SimpleNamespace(
            dataset_metrics={"input": 12},
            is_multi_batch=True,
            context=SimpleNamespace(acquisition_mode="positive"),
        ),
    }
    for name in (
        "raw_result",
        "sample_mv_result",
        "mv_filter_result",
        "correction_result",
        "quality_filter_result",
        "imputation_result",
        "normalization_result",
        "qa_raw_result",
        "qa_mv_result",
        "qa_quality_filter_result",
        "qa_imputation_result",
        "qa_normalization_result",
    ):
        namespace[name] = SimpleNamespace(
            audit=SimpleNamespace(metrics={"label": name})
        )
    namespace["qa_correction_results"] = {
        name: SimpleNamespace(audit=SimpleNamespace(metrics={"label": name}))
        for name in correction_names
    }
    report_source = next(
        "".join(cell["source"])
        for cell in _cells()
        if cell["cell_type"] == "code"
        and "compile_assessor_report(" in "".join(cell["source"])
    )
    exec(compile(report_source, "tutorial report cell", "exec"), namespace)
    call = visual.return_value.compile_assessor_report.call_args
    assert "cleanup_source_svgs" not in call.kwargs
    audits = call.kwargs["audits"]
    assert list(audits) == [
        "Raw data",
        "Missing-value filtering",
        *correction_names,
        "Low-quality filtering",
        "Imputation",
        "Normalization",
    ]
    assert (
        audits[correction_names[0]]
        is namespace["qa_correction_results"][correction_names[0]].audit
    )
    assert namespace["report_input"].asset_manifest == assets
    assert "sample_missing_value_filtering" in namespace["pipeline_metrics"]
    assert (
        namespace["pipeline_metrics"]["raw_dataset"]
        is namespace["raw_result"].audit.metrics
    )
    narrative.return_value.generate_markdown.assert_called_once()
    narrative.return_value.export_report.assert_not_called()


def test_tutorial_explanations_match_selected_methods():
    """Keep walkthrough prose focused on analysis and current method choices."""
    markdown = "\n".join(
        "".join(cell["source"])
        for cell in _cells()
        if cell["cell_type"] == "markdown"
    )
    assert "## Step 02: Missing-Value Filtering\n" in markdown
    assert "two independent actions" not in markdown
    assert 'base_est="Auto"' in markdown
    assert "R-route targets use a fixed reconstruction method" in markdown
    assert "special-handling route (S-route)" in markdown
    assert "not the R programming language" in markdown
    assert "QRILC" in markdown
    assert 'norm_method="Auto"' in markdown
    assert "patchworklib" in markdown


def test_tutorial_initialization_documents_developer_and_runtime_context():
    """Keep the development helper, version, and path style discoverable."""
    code = "\n".join(
        "".join(cell["source"])
        for cell in _cells()
        if cell["cell_type"] == "code"
    )
    assert "%load_ext autoreload" in code
    assert "%autoreload 2" in code
    assert "pimqc.__version__" in code
    initialization = next(
        "".join(cell["source"])
        for cell in _cells()
        if cell["cell_type"] == "code"
        and "pimqc.init(" in "".join(cell["source"])
    )
    assert "os.path.join" in initialization
    assert "Path(" not in initialization


def test_tutorial_respects_loaded_configuration():
    """Use file settings without study-specific parameter or worker caps."""
    for cell in _cells():
        if cell["cell_type"] != "code":
            continue
        for node in ast.walk(ast.parse(_python_source(cell))):
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.ctx, ast.Store)
            ):
                root = node.value
                while isinstance(root, ast.Subscript):
                    root = root.value
                assert not (isinstance(root, ast.Name) and root.id == "params")


def test_tutorial_does_not_depend_on_private_manuscript_exports():
    """Keep the public tutorial independent of author-only figure assembly."""
    source = "\n".join("".join(cell["source"]) for cell in _cells())
    for private_marker in (
        "TEMPORARY ARTICLE DASHBOARD", "EXPORT_ARTICLE_DASHBOARDS",
        "ARTICLE_FIGURE_DIR", "manuscript_figure_drafts.ipynb",
    ):
        assert private_marker not in source
