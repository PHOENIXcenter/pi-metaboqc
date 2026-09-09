"""Tests for the structured report input and Markdown rendering contract."""

import json
from pathlib import Path

import pytest

from pimqc.reporting import NarrativeStatsReporter, ReportInput


def _report_input() -> ReportInput:
    """Return the smallest complete report input accepted by both templates."""
    return ReportInput(
        pipeline_metrics={
            "raw_dataset": {
                "mode": "POS",
                "pi-metaboqc_version": "test-version",
                "features": {"total": 2, "internal_standards": []},
                "samples": {"total": 3, "qc": 1, "blank": 1, "actual": 1},
                "batches": {"batch_count": 1, "batch_distribution": {}},
            },
            "high_mv_feature_filtering": {
                "sample_wise": {
                    "thresholds": {"sample_mv_tol": 0.42},
                    "feature_retention": {
                        "total_checked": 3,
                        "retained_count": 3,
                        "retention_rate_pct": 100.0,
                    },
                },
                "feature_wise": {
                    "filtering_level": "Group",
                    "thresholds": {
                        "mv_group_tol": 0.31,
                        "mv_qc_tol": 0.27,
                        "mnar_group_mv_tol": 0.73,
                        "mnar_qc_mv_tol": 0.19,
                    },
                    "missing_classification": {
                        "mar_count": 1,
                        "mnar_total": 1,
                    },
                    "feature_retention": {
                        "pre_mv_filter_count": 2,
                        "retention_rate_pct": 100.0,
                    },
                },
            },
            "signal_correction": {
                "correction_status": "Completed",
                "overall_performance": {},
                "stages_executed": [
                    {
                        "stage_name": "Global correction",
                        "algorithm": "LOESS",
                        "parameters": {"loess_span": 0.61},
                    }
                ],
            },
            "low_quality_feature_filtering": {
                "feature_retention": {
                    "pre_stage2": {"mar_count": 1, "mnar_count": 1},
                    "post_blank_check": {
                        "mar_count": 1,
                        "mnar_count": 1,
                    },
                    "post_rsd_check": {
                        "mar_count": 1,
                        "mnar_count": 1,
                    },
                },
                "thresholds": {
                    "blank_qc_ratio_tol": 0.17,
                    "qc_rsd_tol": 0.25,
                },
                "filtering_breakdown": {},
            },
            "missing_value_imputation": {
                "imputation_status": "Completed",
                "strategies": {
                    "mnar_method": "row-wise",
                    "mnar_fraction": 0.5,
                },
                "feature_distribution": {"mar_count": 1, "mnar_count": 1},
                "selection": {
                    "requested_method": "KNN",
                    "selected_method": "KNN",
                    "selected_label": "KNN",
                    "is_auto": False,
                    "candidate_results": [
                        {
                            "method": "KNN",
                            "selected": True,
                            "status": "ok",
                            "nrmse_low": 0.12,
                            "nrmse_total": 0.08,
                            "jsd_total": 0.125,
                            "wasserstein_normalized": 0.375,
                        }
                    ],
                },
            },
            "normalization": {
                "strategies": {
                    "normalization_method": "QUANTILE",
                    "log_transform_active": True,
                },
                "selection": {
                    "requested_method": "QUANTILE",
                    "selected_method": "QUANTILE",
                    "selected_label": "QUANTILE",
                    "is_auto": False,
                },
            },
        },
        qa_metrics={
            "raw_dataset": {
                "pca": {"scaling_method": "unit-test-scaling"},
                "correlation": {"method": "pearson"},
            }
        },
        metadata={"date": "2026-08-18 12:00", "mode": "POS"},
        resolved_config={"Dataset": {"mode": "POS"}},
        asset_manifest={"pca": "assets/02_PCA_Scatter_Dashboard.svg"},
    )


def test_report_input_writes_a_portable_json_snapshot(tmp_path: Path) -> None:
    """Persist report state without retaining DataFrame or processor objects."""
    output_path = _report_input().write_json(tmp_path / "Report_Input.json")

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["metadata"]["mode"] == "POS"
    assert payload["resolved_config"]["Dataset"]["mode"] == "POS"
    assert payload["asset_manifest"]["pca"].endswith(".svg")


def test_reporter_renders_markdown_from_report_input(tmp_path: Path) -> None:
    """Use the structured report contract as the only narrative input."""
    reporter = NarrativeStatsReporter(base_dir=str(tmp_path))

    reporter.generate_markdown(_report_input(), report_folder="report")

    report_dir = tmp_path / "report"
    comprehensive = (report_dir / "Report_Comprehensive.md").read_text(
        encoding="utf-8"
    )
    assert (report_dir / "Report_Input.json").is_file()
    assert "Ionization Mode:** POS" in comprehensive
    assert "test-version" in comprehensive
    assert "0.42" in comprehensive
    assert "0.31" in comprehensive
    assert "0.27" in comprehensive
    assert "0.73" in comprehensive
    assert "0.19" in comprehensive
    assert "0.17" in comprehensive
    assert "0.25" in comprehensive
    assert "LOESS" in comprehensive
    assert "KNN" in comprehensive
    assert "QUANTILE" in comprehensive
    assert "unit-test-scaling" in comprehensive
    assert "Pearson" in comprehensive
    assert "MV_Classification_Dashboard.svg" in comprehensive
    assert "Missing-Value Classification Dashboard" in comprehensive
    assert "MV_Feature_Classification_Dashboard.svg" not in comprehensive
    assert "**Imputation of MNAR Features**\n\nFor" in comprehensive
    assert "**Configured MAR Imputation Method**\n\nFor" in comprehensive
    assert "Jensen-Shannon distance of **0.125**" in comprehensive
    assert "normalized Wasserstein distance of **0.375**" in comprehensive


@pytest.mark.parametrize("imputation_status", ["Skipped", "Completed"])
def test_skipped_mv_and_mnar_only_reports_omit_unproduced_figures(
    tmp_path, imputation_status
):
    """Render current audit states even if an older dashboard file exists."""
    report = _report_input()
    metrics = report.pipeline_metrics
    metrics["high_mv_feature_filtering"]["feature_wise"].update(
        {"execution_status": "skipped", "skip_reason": "No missing values."}
    )
    imputation = metrics["missing_value_imputation"]
    imputation["imputation_status"] = imputation_status
    imputation["feature_distribution"] = {"mar_count": 0, "mnar_count": 2}
    reporter = NarrativeStatsReporter(base_dir=str(tmp_path))
    reporter.generate_markdown(report, report_folder="report")
    text = (tmp_path / "report" / "Report_Comprehensive.md").read_text(
        encoding="utf-8"
    )
    assert "MV_Classification_Dashboard.svg" not in text
    assert "Imputation_Dashboard_" not in text
    assert "Imputation_Candidate_Dashboard_" not in text
    assert "Automatic MAR Imputation Selection" not in text


def test_missing_pandoc_never_downloads_or_installs(tmp_path, monkeypatch):
    """Report export is a conversion operation, not environment provisioning."""
    import pypandoc

    source = tmp_path / "Report_Comprehensive.md"
    source.write_text("# Report\n", encoding="utf-8")
    reporter = NarrativeStatsReporter(base_dir=str(tmp_path))
    reporter._generated_md_paths = [str(source)]

    def missing():
        raise OSError("Pandoc unavailable")

    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected download")

    monkeypatch.setattr(pypandoc, "get_pandoc_version", missing)
    monkeypatch.setattr(pypandoc, "download_pandoc", forbidden)
    assert not reporter.export_report()


def test_partial_markdown_generation_cannot_claim_export_success(
    tmp_path, monkeypatch
):
    """An earlier successful report cannot mask a failed current template."""
    reporter = NarrativeStatsReporter(base_dir=str(tmp_path))
    reporter.generate_markdown(_report_input(), report_folder="report")
    original = reporter.env.get_template

    def template(name):
        if "brief" in name:
            raise ValueError("Broken template")
        return original(name)

    monkeypatch.setattr(reporter.env, "get_template", template)
    monkeypatch.setattr(reporter, "_debug_template_errors", lambda *args: None)
    reporter.generate_markdown(_report_input(), report_folder="report")
    assert len(reporter._generated_md_paths) == 1
    assert not reporter.export_report()


def test_all_comprehensive_tables_have_pandoc_captions(
    tmp_path: Path,
) -> None:
    """Keep generated tables in the numbered PDF caption system."""
    report_input = _report_input()
    report_input.pipeline_metrics["signal_correction"]["selection"] = {
        "requested_method": "AUTO",
        "selected_method": "LOESS",
        "selected_label": "LOESS",
        "is_auto": True,
        "candidate_results": [
            {
                "method": "LOESS",
                "selected": True,
                "status": "ok",
                "auto_score": 0.8,
            }
        ],
    }
    report_input.pipeline_metrics["missing_value_imputation"]["selection"][
        "requested_method"
    ] = "AUTO"
    report_input.pipeline_metrics["missing_value_imputation"]["selection"][
        "is_auto"
    ] = True
    report_input.pipeline_metrics["normalization"]["selection"].update(
        {
            "requested_method": "AUTO",
            "is_auto": True,
            "candidate_results": [
                {
                    "method": "QUANTILE",
                    "selected": True,
                    "status": "ok",
                    "overall_score": 0.7,
                }
            ],
        }
    )

    reporter = NarrativeStatsReporter(base_dir=str(tmp_path))
    reporter.generate_markdown(report_input, report_folder="report")
    comprehensive = (tmp_path / "report" / "Report_Comprehensive.md").read_text(
        encoding="utf-8"
    )

    assert "Table: Correction Candidate Comparison" in comprehensive
    assert "Table: MAR Imputation Candidate Comparison" in comprehensive
    assert "Table: Normalization Candidate Comparison" in comprehensive
    lines = comprehensive.splitlines()
    table_count = sum(
        bool(NarrativeStatsReporter._TABLE_SEPARATOR.fullmatch(line))
        for line in lines
    )
    caption_count = sum(line.startswith("Table: ") for line in lines)
    assert table_count == caption_count


def test_export_preflight_rejects_unresolved_assets_and_table_titles(
    tmp_path: Path,
) -> None:
    """Reject blank figures and unnumbered tables before PDF conversion."""
    asset = tmp_path / "figure.svg"
    asset.write_text(
        "<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8"
    )
    markdown = tmp_path / "report.md"
    markdown.write_text(
        "Table: Parsed values\n\n"
        "| Parameter | Value |\n"
        "| --- | --- |\n"
        "| method | AUTO |\n\n"
        "![Dashboard](figure.svg)\n",
        encoding="utf-8",
    )

    NarrativeStatsReporter._validate_markdown_source(markdown)

    asset.unlink()
    with pytest.raises(FileNotFoundError, match="figure.svg"):
        NarrativeStatsReporter._validate_markdown_source(markdown)

    asset.write_text(
        "<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8"
    )
    markdown.write_text(
        "| Parameter | Value |\n"
        "| --- | --- |\n"
        "| method | AUTO |\n\n"
        "![Dashboard](figure.svg)\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="lack a caption"):
        NarrativeStatsReporter._validate_markdown_source(markdown)


def test_imputation_dashboards_follow_selection_evidence(
    tmp_path: Path,
) -> None:
    """Place candidate dashboard before the selected-method dashboard."""
    report_input = _report_input()
    selection = report_input.pipeline_metrics["missing_value_imputation"][
        "selection"
    ]
    selection["requested_method"] = "AUTO"
    selection["is_auto"] = True

    reporter = NarrativeStatsReporter(base_dir=str(tmp_path))
    reporter.generate_markdown(report_input, report_folder="report")

    comprehensive = (tmp_path / "report" / "Report_Comprehensive.md").read_text(
        encoding="utf-8"
    )
    assert comprehensive.index("Masked-Value Distribution Fidelity") < (
        comprehensive.index("MAR Imputation Candidate Comparison")
    )
    assert comprehensive.index("MAR Imputation Candidate Comparison") < (
        comprehensive.index("MAR Imputation Dashboard: KNN")
    )
    assert comprehensive.index("MAR Imputation Candidate Dashboard") < (
        comprehensive.index("MAR Imputation Dashboard: KNN")
    )
    assert "Imputation_Candidate_Dashboard_KNN.svg" in comprehensive
    assert "KNN (selected)" in comprehensive


def test_auto_normalization_renders_candidate_comparison(
    tmp_path: Path,
) -> None:
    """Render a compact AUTO comparison from the unified candidate contract."""
    report_input = _report_input()
    selection = report_input.pipeline_metrics["normalization"]["selection"]
    selection.update(
        {
            "requested_method": "AUTO",
            "is_auto": True,
            "selected_score": 0.71,
            "selection_margin": 0.05,
            "candidate_results": [
                {
                    "method": "ROBUST_LOG_ONLY",
                    "selected": False,
                    "status": "ok",
                    "overall_score": 0.5,
                    "rle_alignment_change_score": 0.5,
                    "variance_stabilization_score": 0.5,
                    "qc_structure_change_score": 0.5,
                    "sample_structure_score": 0.5,
                },
                {
                    "method": "PQN",
                    "selected": True,
                    "status": "ok",
                    "overall_score": 0.71,
                    "rle_alignment_change_score": 1.0,
                    "variance_stabilization_score": 0.5,
                    "qc_structure_change_score": 0.86,
                    "sample_structure_score": 0.46,
                },
            ],
        }
    )

    reporter = NarrativeStatsReporter(base_dir=str(tmp_path))
    reporter.generate_markdown(report_input, report_folder="report")

    comprehensive = (tmp_path / "report" / "Report_Comprehensive.md").read_text(
        encoding="utf-8"
    )
    assert "Normalization Candidate Comparison" in comprehensive
    assert "PQN (selected)" in comprehensive


def test_reporter_does_not_render_missing_metrics_as_zero(
    tmp_path: Path,
) -> None:
    """Show unavailable correction and VSN metrics without numeric fallbacks."""
    report_input = _report_input()
    report_input.pipeline_metrics["signal_correction"][
        "stages_executed"
    ].append(
        {
            "stage_name": "Final correction",
            "algorithm": "LOESS",
            "parameters": {},
        }
    )
    report_input.pipeline_metrics["normalization"]["strategies"][
        "normalization_method"
    ] = "VSN"

    reporter = NarrativeStatsReporter(base_dir=str(tmp_path))
    reporter.generate_markdown(report_input, report_folder="report")

    report_dir = tmp_path / "report"
    brief = (report_dir / "Report_Brief.md").read_text(encoding="utf-8")
    comprehensive = (report_dir / "Report_Comprehensive.md").read_text(
        encoding="utf-8"
    )

    assert "QC RSD metrics are unavailable" in brief
    assert "without a reportable QC RSD summary" in comprehensive
    assert "0.000e+00" not in comprehensive
    assert "structural scale factor of **N/A**" in comprehensive
