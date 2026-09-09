"""Report narrative, visual-asset assembly, and document-rendering utilities.

The module collects stage figures and metrics, constructs scientific narrative
statistics, draws audit-backed QA dashboards, renders Jinja report templates,
and exports Markdown, HTML, or PDF deliverables. It also validates template
references and removes intermediate assets after report assembly when required.
"""

import os
import re
from datetime import datetime
import sys

import subprocess
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from tabulate import tabulate

from loguru import logger

from .models import ReportInput
from typing import Union, Optional, Dict, Any

# from ..plotting.assembly import stitch_svg_grids as stitch_svg_grids
from collections.abc import Mapping
from ..processing.audit import AssessQualityAuditPayload
from ..plotting.assessment.comparison import render_assessment_comparison


# =============================================================================
# Visual Asset Reporter
# =============================================================================
class VisualAssetReporter:
    """Collect and arrange pipeline visual assets for report generation."""

    def __init__(self, base_dir: Union[str, Path]) -> None:
        """
        Initialize the reporter at the project workspace level.

        Args:
            base_dir: Root output directory containing step folders.
        """
        self.base_dir = Path(base_dir)
        self.qa_folders = self._detect_qa_folders()

    def _detect_qa_folders(self) -> list[str]:
        """Automatically scan for QA directories and sort them."""
        if not self.base_dir.exists():
            return []
        folders = [
            d for d in self.base_dir.iterdir() if d.is_dir() and "QA" in d.name
        ]
        return sorted([d.name for d in folders])

    def compile_assessor_report(
        self,
        audits: Mapping[str, AssessQualityAuditPayload],
        is_multi_batch: bool = True,
        report_folder: str = "13_Report_Markdown",
        cols: Union[int, str] = "auto",
        show_plot: bool = True,
    ) -> dict[str, str]:
        """
        Draw QA grids from explicit audits using the shared patchwork renderer.
        """
        report_path = self.base_dir / report_folder
        assets = render_assessment_comparison(
            audits,
            report_path / "assets",
            is_multi_batch=is_multi_batch,
            cols=cols,
            show_plot=show_plot,
        )
        return {
            name: str(path.relative_to(report_path))
            for name, path in assets.items()
        }


# =============================================================================
# Narrative Statistics Reporter
# =============================================================================
class NarrativeStatsReporter:
    """Extract metadata from structured stage metrics for one report."""

    _IMAGE_REFERENCE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
    _TABLE_SEPARATOR = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*$")

    # --- Define CSS as a class constant for cleaner maintenance ---
    REPORT_CSS = """
    /* ====================================================================
     * PI-METABOQC UNIFIED REPORT STYLESHEET
     * Optimized for WeasyPrint (PDF) and modern web browsers (HTML).
     * ==================================================================== */

    /* --- 1. Screen Reading Layout (Web Browsers) --- */
    @media screen {
        body {
            max-width: 1200px !important;
            margin: 0 auto !important;
            padding: 20px;
        }
    }

    /* --- 2. Default Image Constraints --- */
    .full-width-image {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 100%;
        height: auto;
    }

    /* --- 3. Paged Layout (PDF only; Markdown remains portable) --- */
    @media print {
        @page {
            size: A4;
            margin: 20mm;
        }

        body {
            /* Override Pandoc's screen-oriented body width and padding. */
            max-width: none;
            margin: 0;
            padding: 0;
            font-size: 11pt;
            line-height: 1.3;
        }

        h1, h2, h3, h4, h5, h6 {
            break-after: avoid;
            page-break-after: avoid;
            break-inside: avoid;
        }

        p, li {
            orphans: 3;
            widows: 3;
        }

        body figure {
            break-inside: avoid;
            page-break-inside: avoid;
            margin: 10pt 0 12pt;
        }

        figure .full-width-image {
            /* Auto dimensions preserve SVG aspect ratio under BOTH limits.
             * Reserve vertical room for a multi-line caption on A4. */
            width: auto;
            height: auto;
            max-width: 100%;
            max-height: 205mm;
            margin: 0 auto;
        }

        body figcaption {
            margin-top: 6pt;
            font-size: 9pt;
            line-height: 1.25;
            break-before: avoid;
        }

        body table {
            max-width: 100%;
            margin: 8pt auto 12pt;
            font-size: 8.5pt;
            line-height: 1.25;
            /* Long tables may span pages; individual rows stay together. */
            break-inside: auto;
        }

        body th, body td {
            padding: 4pt 6pt;
            overflow-wrap: break-word;
        }

        body thead {
            display: table-header-group;
        }

        body tr {
            break-inside: avoid;
            page-break-inside: avoid;
        }

        body caption {
            font-size: 9pt;
            margin-bottom: 6pt;
            break-after: avoid;
            page-break-after: avoid;
        }
    }

    /* --- 4. Academic Three-Line Table Style --- */
    table {
        border-collapse: collapse;
        width: auto;      /* Let the table shrink to fit its content */
        min-width: 80%;   /* Maintain a reasonably wide academic look */
        margin: 0 auto 24px auto; /* Center table and add bottom spacing */
        font-size: 14px;
    }
    th, td {
        padding: 8px 12px;
        text-align: left;
    }
    thead {
        border-top: 2px solid black;
        border-bottom: 1px solid black;
    }
    tbody {
        border-bottom: 2px solid black;
    }

    /* --- 5. Figure and Table Captions --- */
    figure {
        /* Remove default margin, keep bottom spacing */
        margin: 0 0 24px 0; 
        /* Ensure the containing block is centered */
        text-align: center; 
    }
    figcaption {
        text-align: center;
        font-size: 14px;
        color: #333333;
        margin-top: 10px;
        font-weight: 500;
    }
    caption {
        caption-side: top;  /* Force table title above the table */
        text-align: center; /* Center the title */
        font-weight: bold;
        color: #333333;
        margin-bottom: 8px; /* Spacing between title and table */
    }

    /* --- 6. CSS Auto-numbering Magic --- */
    body {
        /* Initialize chapter counter. (Assuming chapter 2 for QA) */
        counter-reset: chapter-counter 0; 
    }
    h2 {
        /* Increment chapter counter, reset table and figure counters */
        counter-increment: chapter-counter;
        counter-reset: table-counter figure-counter;
    }
    caption::before {
        /* Auto-generate Table prefix (e.g., "Table 2.1: ") */
        counter-increment: table-counter;
        content: "Table " counter(chapter-counter) "." 
                counter(table-counter) ": ";
    }
    figcaption::before {
        /* Auto-generate Figure prefix (e.g., "Figure 2.1: ") */
        counter-increment: figure-counter;
        content: "Figure " counter(chapter-counter) "." 
                counter(figure-counter) ": ";
        font-weight: bold;
    }

    """
    # Define standard pipeline stages for ordered iteration
    _QA_STAGES = [
        ("raw_dataset", "Raw data"),
        ("high_mv_feature_filtering", "High-missing value features filtering"),
        ("intra_batch_correction", "Intra-batch correction"),
        ("inter_batch_correction", "Inter-batch correction"),
        ("global_correction", "Global model correction"),
        ("low_quality_feature_filtering", "Low-quality features filtering"),
        ("missing_value_imputation", "Imputation"),
        ("normalization", "Normalization"),
    ]

    def __init__(self, base_dir: str) -> None:
        """Initialize the reporter with base directory and Jinja2 env."""
        self.base_dir = Path(base_dir)
        template_path = Path(__file__).resolve().parents[1] / "templates"
        if not template_path.is_dir():
            raise FileNotFoundError(
                f"Package templates directory is missing: {template_path}"
            )
        self.env = Environment(loader=FileSystemLoader(str(template_path)))

    @classmethod
    def _validate_markdown_source(cls, markdown_path: Path) -> None:
        """Fail before export when assets or table captions are unresolved."""
        content = markdown_path.read_text(encoding="utf-8")
        missing_assets: list[str] = []
        for match in cls._IMAGE_REFERENCE.finditer(content):
            target = match.group(1).strip()
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1]
            else:
                target = target.split(maxsplit=1)[0]
            if target.startswith(("http://", "https://", "data:", "#")):
                continue
            asset_path = markdown_path.parent / Path(target.replace("\\", "/"))
            if not asset_path.is_file():
                missing_assets.append(target)

        if missing_assets:
            missing = ", ".join(sorted(set(missing_assets)))
            raise FileNotFoundError(
                f"Unresolved report image reference(s): {missing}"
            )

        lines = content.splitlines()
        uncaptioned_tables: list[str] = []
        for index, line in enumerate(lines):
            if not cls._TABLE_SEPARATOR.fullmatch(line):
                continue
            header_index = index - 1
            while header_index >= 0 and not lines[header_index].strip():
                header_index -= 1
            caption_index = header_index - 1
            while caption_index >= 0 and not lines[caption_index].strip():
                caption_index -= 1
            if caption_index < 0 or not lines[caption_index].startswith(
                "Table: "
            ):
                header = lines[header_index].strip()
                uncaptioned_tables.append(header)

        if uncaptioned_tables:
            headers = ", ".join(uncaptioned_tables)
            raise ValueError(f"Report table(s) lack a caption: {headers}")

        caption_count = sum(
            line.startswith("Table: ") for line in content.splitlines()
        )
        table_count = sum(
            bool(cls._TABLE_SEPARATOR.fullmatch(line))
            for line in content.splitlines()
        )
        if caption_count != table_count:
            raise ValueError(
                "Report table-caption count mismatch: "
                f"{table_count} table(s), {caption_count} caption(s)."
            )

    def _create_batch_table(self, batch_dist: Dict[str, Any]) -> str:
        """Generates a Markdown table displaying batch sample distributions."""
        rows = []
        sum_total = 0
        sum_qc = 0
        sum_blank = 0
        sum_sample = 0

        if isinstance(batch_dist, dict):
            for b_id, b_info in batch_dist.items():
                total = b_info.get("Total", 0)
                qc = b_info.get("QC", 0)
                blank = b_info.get("Blank", 0)
                sample = b_info.get("Sample", 0)

                sum_total += total
                sum_qc += qc
                sum_blank += blank
                sum_sample += sample

                rows.append(
                    [
                        b_id,
                        total,
                        qc,
                        blank,
                        sample,
                        b_info.get("Inject Order", "N/A"),
                    ]
                )

        if rows:
            rows.append(["All", sum_total, sum_qc, sum_blank, sum_sample, "/"])

        headers = ["Batch", "Total", "QC", "Blank", "Sample", "Inject Order"]
        table_str = tabulate(
            rows,
            headers=headers,
            tablefmt="github",
            stralign="center",
            numalign="center",
        )
        return f"\n\n{table_str}\n\n"

    def _create_rsd_summary_table(self, qa_metrics: Dict[str, Any]) -> str:
        """Generates a table detailing QC and Sample RSD distribution.

        Args:
            qa_metrics (Disct[str, Any]): Dictionary of quality assessment
                outputs for all pipeline stages.

        Returns:
            str: A formatted Markdown table string.
        """
        rows = []
        for stage_key, stage_name in self._QA_STAGES:
            qa_data = qa_metrics.get(stage_key, {})
            if not qa_data:
                continue

            rsd_dist = qa_data.get("rsd_distribution", {})
            if not rsd_dist:
                continue

            # Extract metrics for both Quality Control and Actual Samples
            qc_rsd = rsd_dist.get("qc", {})
            sample_rsd = rsd_dist.get("actual", {})

            # Build matrix rows with nested group logic for scannability
            if qc_rsd or sample_rsd:
                rows.append(
                    [
                        stage_name,
                        # QC Data
                        qc_rsd.get("0-10%", 0),
                        qc_rsd.get("10-20%", 0),
                        qc_rsd.get("20-30%", 0),
                        qc_rsd.get(">30%", 0),
                        # Sample Data
                        sample_rsd.get("0-10%", 0),
                        sample_rsd.get("10-20%", 0),
                        sample_rsd.get("20-30%", 0),
                        sample_rsd.get(">30%", 0),
                    ]
                )

        headers = [
            "Pipeline Stage",
            "QC 0-10%",
            "QC 10-20%",
            "QC 20-30%",
            "QC >30%",
            "Sample 0-10%",
            "Sample 10-20%",
            "Sample 20-30%",
            "Sample >30%",
        ]

        table_str = tabulate(
            rows,
            headers=headers,
            tablefmt="github",
            stralign="center",
            numalign="center",
        )
        return f"\n\n{table_str}\n\n" if rows else ""

    def _summarize_rsd_guardrail(
        self, qa_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Summarize QC precision and biological-sample RSD distribution
        separately.

        Pooled-QC RSD is an analytical-precision measure. In contrast, RSD
        across
        actual biological samples is retained as a descriptive guardrail: it
            should
        remain broad after preprocessing rather than being minimized as though
        it
        were purely technical variation.

        """
        labels = ("0-10%", "10-20%", "20-30%", ">30%")

        def get_distribution(
            stage_key: str, sample_class: str
        ) -> Dict[str, int]:
            stage_data = qa_metrics.get(stage_key, {})
            distribution = stage_data.get("rsd_distribution", {})
            values = distribution.get(sample_class, {})
            return {label: int(values.get(label, 0)) for label in labels}

        raw_qc = get_distribution("raw_dataset", "qc")
        raw_actual = get_distribution("raw_dataset", "actual")
        raw_total = sum(raw_qc.values())
        raw_actual_total = sum(raw_actual.values())

        final_key = ""
        final_name = ""
        for stage_key, stage_name in reversed(self._QA_STAGES):
            if qa_metrics.get(stage_key, {}).get("rsd_distribution"):
                final_key = stage_key
                final_name = stage_name
                break

        if not final_key or raw_total == 0 or raw_actual_total == 0:
            return {"available": False}

        final_qc = get_distribution(final_key, "qc")
        final_actual = get_distribution(final_key, "actual")
        final_total = sum(final_qc.values())
        actual_total = sum(final_actual.values())
        if final_total == 0 or actual_total == 0:
            return {"available": False}

        return {
            "available": True,
            "final_stage": final_name,
            "raw_qc_low_count": raw_qc["0-10%"],
            "raw_qc_total": raw_total,
            "raw_qc_low_pct": 100 * raw_qc["0-10%"] / raw_total,
            "final_qc_low_count": final_qc["0-10%"],
            "final_qc_total": final_total,
            "final_qc_low_pct": 100 * final_qc["0-10%"] / final_total,
            "raw_actual_high_count": raw_actual[">30%"],
            "raw_actual_total": raw_actual_total,
            "raw_actual_high_pct": 100 * raw_actual[">30%"] / raw_actual_total,
            "final_actual_high_count": final_actual[">30%"],
            "final_actual_total": actual_total,
            "final_actual_high_pct": 100 * final_actual[">30%"] / actual_total,
        }

    def _create_pca_summary_table(self, qa_metrics: Dict[str, Any]) -> str:
        """Generates a table detailing PCA drift and silhouette metrics."""
        rows = []
        for stage_key, stage_name in self._QA_STAGES:
            qa_data = qa_metrics.get(stage_key, {})
            if not qa_data:
                continue
            pca = qa_data.get("pca", {})
            if pca:
                pc1 = pca.get("pc1_variance")
                pc1_str = f"{pc1 * 100:.2f}%" if pc1 else "N/A"

                pc2 = pca.get("pc2_variance")
                pc2_str = f"{pc2 * 100:.2f}%" if pc2 else "N/A"

                disp = pca.get("relative_dispersion")
                disp_str = f"{disp:.4f}" if disp is not None else "N/A"

                silh = pca.get("batch_silhouette")
                silh_str = f"{silh:.4f}" if silh is not None else "N/A"

                shift = pca.get("centrality_shift")
                shift_str = f"{shift:.4f}" if shift is not None else "N/A"

                rows.append(
                    [
                        stage_name,
                        pc1_str,
                        pc2_str,
                        disp_str,
                        silh_str,
                        shift_str,
                    ]
                )

        headers = [
            "Pipeline Stage",
            "PC1 Var",
            "PC2 Var",
            "Rel. Dispersion",
            "Batch Silh.",
            "Cent. Shift",
        ]
        table_str = tabulate(
            rows,
            headers=headers,
            tablefmt="github",
            disable_numparse=True,
        )
        return f"\n\n{table_str}\n\n" if rows else ""

    def _create_corr_summary_table(self, qa_metrics: Dict[str, Any]) -> str:
        """Generates a table summarizing pooled QC correlations.
        Dynamically adapts columns based on whether the dataset is single
        or multi-batch.
        """
        # Determine if the dataset is multi-batch by scanning QA metrics
        is_multi_batch = False
        for stage_key, _ in self._QA_STAGES:
            qa_data = qa_metrics.get(stage_key, {})
            if (
                qa_data.get("correlation", {})
                .get("batch_level", {})
                .get("is_multi_batch", False)
            ):
                is_multi_batch = True
                break

        rows = []
        for stage_key, stage_name in self._QA_STAGES:
            qa_data = qa_metrics.get(stage_key, {})
            if not qa_data:
                continue

            corr_data = qa_data.get("correlation", {})
            sample_level = corr_data.get("sample_level", {})
            batch_level = corr_data.get("batch_level", {})

            # Extract inner-batch median (with fallback to legacy "median" key)
            inner = sample_level.get("inner_batch_median", "N/A")
            if inner == "N/A":
                inner = corr_data.get("median", "N/A")

            # Strictly format to 4 decimal places to ensure alignment
            inner_str = (
                f"{inner:.4f}" if isinstance(inner, float) else str(inner)
            )

            row = [stage_name, inner_str]

            # Append multi-batch specific metrics only if applicable
            if is_multi_batch:
                cross = sample_level.get("cross_batch_median", "N/A")
                worst_pair = batch_level.get("worst_batch_pair", "N/A")
                worst_corr = batch_level.get("worst_correlation", "N/A")

                cross_str = (
                    f"{cross:.4f}" if isinstance(cross, float) else str(cross)
                )
                worst_corr_str = (
                    f"{worst_corr:.4f}"
                    if isinstance(worst_corr, float)
                    else str(worst_corr)
                )

                row.extend([cross_str, worst_pair, worst_corr_str])

            rows.append(row)

        # Dynamically set headers based on batch design
        if is_multi_batch:
            headers = [
                "Pipeline Stage",
                "Inner-Batch Median",
                "Cross-Batch Median",
                "Worst Batch Pair",
                "Worst Batch Corr.",
            ]
        else:
            headers = ["Pipeline Stage", "Median Correlation"]

        # Render table with disable_numparse=True to preserve trailing zeros
        table_str = tabulate(
            rows,
            headers=headers,
            tablefmt="github",
            disable_numparse=True,
            stralign="center",
            numalign="center",
        )
        return f"\n\n{table_str}\n\n" if rows else ""

    def _create_outlier_summary_table(self, qa_metrics: Dict[str, Any]) -> str:
        """
        Generates a table detailing multi-dimensional outliers across stages.
        """
        rows = []
        for stage_key, stage_name in self._QA_STAGES:
            q_data = qa_metrics.get(stage_key, {})
            if not q_data:
                continue

            # SD-OD Extreme Outliers
            outliers = q_data.get("outliers", {})
            ext_samples = outliers.get("extreme_samples", [])
            ext_str = (
                ", ".join(map(str, ext_samples))
                if ext_samples
                else "None detected"
            )

            # IS Outliers
            is_qc = q_data.get("internal_standard_qc", {})
            is_samples = is_qc.get("is_outlier_samples", [])
            is_rate = is_qc.get("is_outlier_standard", "Not configured")
            is_str = (
                ", ".join(map(str, is_samples))
                if is_samples
                else "None detected"
                if is_qc
                else "Not configured"
            )

            # ORF Outliers
            orf_qc = q_data.get("orf_qc", {})
            orf_samples = orf_qc.get("orf_outlier_samples", [])
            orf_rate = orf_qc.get("orf_outlier_standard", "Not configured")
            orf_str = (
                ", ".join(map(str, orf_samples))
                if orf_samples
                else "None detected"
                if orf_qc
                else "Not configured"
            )

            rows.append(
                [stage_name, ext_str, is_rate, is_str, orf_rate, orf_str]
            )

        headers = [
            "Pipeline Stage",
            "SD-OD Extreme Outlier Samples",
            "IS Outliers (N/Total)",
            "IS Outlier Samples",
            "ORF Outliers (N/Total)",
            "ORF Outlier Samples",
        ]

        table_str = tabulate(
            rows,
            headers=headers,
            tablefmt="github",
            stralign="center",
            numalign="center",
        )
        return f"\n\n{table_str}\n\n" if rows else ""

    @staticmethod
    def _format_candidate_metric(value: Any) -> str:
        """Format candidate scores without replacing missing values by zero."""
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return f"{value:.3f}"
        return "N/A" if value in (None, "") else str(value)

    def _create_imputation_candidate_table(
        self, pipeline_metrics: Dict[str, Any]
    ) -> str:
        """Render the structured AUTO imputation candidate comparison."""
        stage = pipeline_metrics.get("missing_value_imputation", {})
        selection = stage.get("selection", {})
        requested = str(selection.get("requested_method", ""))
        is_auto = selection.get("is_auto", requested.upper() == "AUTO")
        candidates = selection.get("candidate_results", [])
        if not is_auto or not isinstance(candidates, list) or not candidates:
            return ""

        selected = selection.get("selected_method")
        rows = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            method = candidate.get("method", "N/A")
            label = str(method)
            if candidate.get("selected") or method == selected:
                label += " (selected)"
            rows.append(
                [
                    label,
                    self._format_candidate_metric(candidate.get("nrmse_low")),
                    self._format_candidate_metric(candidate.get("nrmse_total")),
                    self._format_candidate_metric(candidate.get("jsd_total")),
                    self._format_candidate_metric(
                        candidate.get("wasserstein_normalized")
                    ),
                    self._format_candidate_metric(candidate.get("auto_score")),
                ]
            )

        headers = [
            "Candidate",
            "Low NRMSE",
            "Total NRMSE",
            "JSD",
            "Norm. Wasserstein",
            "AUTO score",
        ]
        table = tabulate(
            rows,
            headers=headers,
            tablefmt="github",
            disable_numparse=True,
        )
        return f"\n\n{table}\n\n"

    def _create_normalization_candidate_table(
        self, pipeline_metrics: Dict[str, Any]
    ) -> str:
        """Render the structured AUTO normalization candidate comparison."""
        stage = pipeline_metrics.get("normalization", {})
        selection = stage.get("selection", {})
        requested = str(selection.get("requested_method", ""))
        is_auto = selection.get("is_auto", requested.upper() == "AUTO")
        candidates = selection.get("candidate_results", [])
        if not is_auto or not isinstance(candidates, list) or not candidates:
            return ""

        rows = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            method = str(candidate.get("method", "N/A")).replace("_", " ")
            if candidate.get("selected"):
                method += " (selected)"
            rows.append(
                [
                    method,
                    self._format_candidate_metric(
                        candidate.get("overall_score")
                    ),
                    self._format_candidate_metric(
                        candidate.get("rle_alignment_change_score")
                    ),
                    self._format_candidate_metric(
                        candidate.get("variance_stabilization_score")
                    ),
                    self._format_candidate_metric(
                        candidate.get("qc_structure_change_score")
                    ),
                    self._format_candidate_metric(
                        candidate.get("sample_structure_score")
                    ),
                ]
            )

        if not rows:
            return ""
        headers = [
            "Candidate",
            "Overall",
            "RLE",
            "Variance",
            "QC structure",
            "Sample structure",
        ]
        table = tabulate(
            rows,
            headers=headers,
            tablefmt="github",
            disable_numparse=True,
        )
        return f"\n\n{table}\n\n"

    def _create_correction_candidate_table(
        self, pipeline_metrics: Dict[str, Any]
    ) -> str:
        """Render the structured AUTO correction candidate comparison."""
        stage = pipeline_metrics.get("signal_correction", {})
        selection = stage.get("selection", {})
        requested = str(selection.get("requested_method", ""))
        is_auto = selection.get("is_auto", requested.upper() == "AUTO")
        candidates = selection.get("candidate_results", [])
        if not is_auto or not isinstance(candidates, list) or not candidates:
            return ""

        rows = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            method = str(candidate.get("method", "N/A"))
            if candidate.get("selected"):
                method += " (selected)"
            rows.append(
                [
                    method,
                    self._format_candidate_metric(candidate.get("auto_score")),
                    self._format_candidate_metric(candidate.get("eval_rsd")),
                    self._format_candidate_metric(
                        candidate.get("median_qc_rsd_improvement_score")
                    ),
                    self._format_candidate_metric(
                        candidate.get("featurewise_qc_rsd_improvement_score")
                    ),
                    self._format_candidate_metric(
                        candidate.get("sample_structure_score")
                    ),
                ]
            )

        if not rows:
            return ""
        headers = [
            "Candidate",
            "AUTO score",
            "Eval. QC RSD",
            "Median RSD gain",
            "Feature RSD gain",
            "Sample structure",
        ]
        table = tabulate(
            rows,
            headers=headers,
            tablefmt="github",
            disable_numparse=True,
        )
        return f"\n\n{table}\n\n"

    def consolidate_metrics(self, report_input: ReportInput) -> Dict[str, Any]:
        """Consolidates pipeline and QA metrics into a unified context."""

        pipeline_metrics = report_input.pipeline_metrics
        qa_metrics = report_input.qa_metrics

        def get_val(
            d: Dict[str, Any], *keys: str, default: object = "N/A"
        ) -> object:
            """Safely traverse nested dictionaries to retrieve values."""
            for k in keys:
                if isinstance(d, dict) and k in d:
                    d = d[k]
                else:
                    return default
            return d

        batch_count = get_val(
            pipeline_metrics,
            "raw_dataset",
            "batches",
            "batch_count",
            default=None,
        )

        stats = {
            "metadata": dict(report_input.metadata),
            "resolved_config": dict(report_input.resolved_config),
            "asset_manifest": dict(report_input.asset_manifest),
        }
        stats["metadata"].setdefault(
            "date", datetime.now().strftime("%Y-%m-%d %H:%M")
        )
        stats["metadata"].setdefault(
            "version",
            get_val(
                pipeline_metrics,
                "raw_dataset",
                "pi-metaboqc_version",
                default="Unknown",
            ),
        )
        stats["metadata"].setdefault(
            "mode",
            get_val(pipeline_metrics, "raw_dataset", "mode", default="N/A"),
        )
        is_multi_batch = (
            isinstance(batch_count, (int, float)) and batch_count > 1
        )
        stats["metadata"].setdefault("is_multi_batch", is_multi_batch)

        # [REFACTOR]: Core stages iteration fully handles new 2-Stage Norm logic
        for stage_key, _ in self._QA_STAGES:
            pipe_data = pipeline_metrics.get(stage_key, {})
            if isinstance(pipe_data, dict):
                pipe_data = dict(pipe_data)
                pipeline_metrics[stage_key] = pipe_data
            qa_data = qa_metrics.get(stage_key, {})
            stats[stage_key] = {
                "pipeline_params": pipe_data,
                "qa_assessments": qa_data,
            }
        if "signal_correction" in pipeline_metrics:
            stats["signal_correction"] = pipeline_metrics["signal_correction"]

        batch_dist = get_val(
            pipeline_metrics,
            "raw_dataset",
            "batches",
            "batch_distribution",
            default={},
        )

        stats["raw_dataset"]["batch_table"] = self._create_batch_table(
            batch_dist
        )

        mar_sel = get_val(
            pipeline_metrics,
            "missing_value_imputation",
            "selection",
            "selected_method",
            default="",
        )
        if mar_sel and mar_sel != "N/A":
            candidate_rows = get_val(
                pipeline_metrics,
                "missing_value_imputation",
                "selection",
                "candidate_results",
                default=[],
            )
            selected_row = next(
                (
                    row
                    for row in candidate_rows
                    if isinstance(row, dict) and row.get("method") == mar_sel
                ),
                {},
            )
            nrmse = selected_row.get("nrmse_low", "N/A")
            stats["missing_value_imputation"]["best_nrmse_low"] = nrmse

        stats["summary_tables"] = {
            "rsd": self._create_rsd_summary_table(qa_metrics),
            "pca": self._create_pca_summary_table(qa_metrics),
            "correlation": self._create_corr_summary_table(qa_metrics),
            "outliers": self._create_outlier_summary_table(qa_metrics),
            "correction_candidates": self._create_correction_candidate_table(
                pipeline_metrics
            ),
            "imputation_candidates": self._create_imputation_candidate_table(
                pipeline_metrics
            ),
            "normalization_candidates": (
                self._create_normalization_candidate_table(pipeline_metrics)
            ),
        }
        stats["rsd_guardrail"] = self._summarize_rsd_guardrail(qa_metrics)

        return stats

    def _debug_template_errors(self, template_name: str, context: dict) -> None:
        """
        Advanced debugger to identify template rendering issues without
        triggering false positives from Jinja2's static AST parser.
        """
        import jinja2
        from jinja2 import meta
        import traceback

        logger.info(f"--- Starting Template Debugger for '{template_name}' ---")

        # Static Analysis (Informational only, downgraded to DEBUG level)
        try:
            template_src = self.env.loader.get_source(self.env, template_name)[
                0
            ]
            parsed_content = self.env.parse(template_src)
            ref_vars = meta.find_undeclared_variables(parsed_content)

            missing_top = [var for var in ref_vars if var not in context]
            if missing_top:
                # Static AST analysis cannot fully resolve conditionals ({% if
                # %})
                # or local assignments ({% set %}). Therefore, it often
                # generates
                # false positives. Logging as debug information only.
                logger.debug(
                    "Static AST found potential undeclared variables "
                    f"(often false positives): {missing_top}"
                )
        except Exception as e:
            logger.debug(
                f"Static analysis skipped due to parser limitation: {e}"
            )

        # Runtime Analysis (The Ultimate Source of Truth)
        # Create an isolated, strict environment. Any truly undefined variable
        # evaluated at runtime will instantly trigger an exception.
        strict_env = jinja2.Environment(
            loader=self.env.loader,
            undefined=jinja2.StrictUndefined,  # Enforce strict evaluation
        )

        try:
            debug_template = strict_env.get_template(template_name)

            # Attempt a full render. Success here guarantees the template
            # logic is perfectly sound given the current context.
            debug_template.render(context)
            logger.success(
                f"Template '{template_name}' successfully passed strict "
                "runtime rendering."
            )

        except jinja2.exceptions.UndefinedError as e:
            # Catch actual runtime undefined variables (e.g., typos in keys)
            logger.error(f"RUNTIME UNDEFINED ERROR in '{template_name}': {e}")

        except jinja2.exceptions.TemplateSyntaxError as e:
            # Catch syntax errors (e.g., missing {% endif %})
            logger.error(
                f"SYNTAX ERROR in '{template_name}' at line "
                f"{e.lineno}: {e.message}"
            )

        except Exception as e:
            # Catch other runtime execution errors (e.g., type mismatches)
            logger.error(f"EXECUTION ERROR in '{template_name}': {e}")
            logger.debug(traceback.format_exc())

        logger.info("--- Template Debugger Finished ---")

    def _is_weasyprint_operational(self) -> bool:
        """Performs a hard check to verify if WeasyPrint C-libraries exist."""
        if sys.platform == "win32":
            gtk_bin = os.path.join(sys.prefix, "Library", "bin")
            if os.path.exists(gtk_bin) and (
                gtk_bin not in os.environ.get("PATH", "")
            ):
                os.environ["PATH"] = (
                    f"{gtk_bin}{os.pathsep}{os.environ.get('PATH', '')}"
                )

        try:
            result = subprocess.run(
                ["weasyprint", "--version"],
                timeout=10,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode != 0:
                logger.debug(
                    f"WeasyPrint probe failed. Stderr: {result.stderr.strip()}"
                )
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def _is_xelatex_available(self) -> bool:
        """Perform a bounded operational check of the actual PDF engine.

        Bypasses shutil.which to avoid false positives from broken paths or
        ghost registry entries.

        Returns:
            bool: True if the binary executes successfully, False otherwise.
        """
        try:
            result = subprocess.run(
                ["xelatex", "--version"],
                timeout=10,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            # Only return True if the process exits without error
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            # OSError catches cases where the file exists but not executable
            return False

    def _is_rsvg_operational(self) -> bool:
        """Performs a hard check to verify if rsvg-convert is installed.

        Pandoc requires this system binary to convert SVGs to PDFs on-the-fly
        when targeting XeLaTeX.

        Returns:
            bool: True if the binary executes successfully.
        """
        try:
            result = subprocess.run(
                ["rsvg-convert", "--version"],
                timeout=10,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def generate_markdown(
        self,
        report_input: ReportInput,
        report_folder: str = "08_Report_Summary",
    ) -> None:
        """
        Renders both comprehensive and brief Markdown QC reports.

        Iterates through defined template versions and generates distinct
        markdown documents based on the unified metrics context.
        """
        self._generated_md_paths = []
        self._markdown_generation_failed = True
        # Consolidate double-source data
        context = self.consolidate_metrics(report_input)
        out_dir = self.base_dir / report_folder

        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except IOError as e:
            logger.error(f"Failed to create report directory: {e}")
            return

        report_input.write_json(out_dir / "Report_Input.json")

        # Initialize a list to track all generated markdown paths
        self._generated_md_paths = []

        # Define the template mapping for dual output
        versions = {
            "Comprehensive": "report_comprehensive.md.j2",
            "Brief": "report_brief.md.j2",
        }

        # Loop through versions and render
        for label, template_name in versions.items():
            logger.info(f"Generating {label.upper()} narrative report...")
            try:
                template = self.env.get_template(template_name)
                content = template.render(context)

                md_path = out_dir / f"Report_{label}.md"
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(content)

                self._generated_md_paths.append(md_path)
                logger.success(
                    f"{label.capitalize()} report generated: {md_path}"
                )
            except Exception as e:
                logger.error(f"Jinja2 rendering failed for {label}: {e}")
                self._debug_template_errors(template_name, context)

        self._markdown_generation_failed = len(self._generated_md_paths) != len(
            versions
        )

    def export_report(self, pdf_engine: Optional[str] = "weasyprint") -> bool:
        """Exports all generated markdown reports to PDF/HTML sequentially.

        Orchestrates the conversion from Markdown to PDF/HTML using Pandoc.
        Loops through all tracked markdown files and handles fallbacks.
        """
        md_paths = getattr(self, "_generated_md_paths", [])
        if getattr(self, "_markdown_generation_failed", False):
            logger.error(
                "The current run did not generate both report templates."
            )
            return False
        if not md_paths:
            logger.error("No Markdown found. Run `generate_markdown` first.")
            return False

        try:
            for md_path in md_paths:
                self._validate_markdown_source(Path(md_path))
        except (FileNotFoundError, ValueError) as error:
            logger.error(f"Report export preflight failed: {error}")
            return False

        # --- Phase 0: Pandoc Environment Check ---
        try:
            import pypandoc

            try:
                pypandoc.get_pandoc_version()
            except OSError:
                logger.error("Pandoc is unavailable; install it explicitly.")
                return False
        except ImportError:
            logger.error("Missing dependency: install pypandoc and Pandoc.")
            return False

        # --- Phase 1: Setup runtime CSS file ---
        # Derive base directory from the first available markdown file
        md_dir = os.path.abspath(os.path.dirname(str(md_paths[0])))
        assets_path = Path(md_dir) / "assets"
        assets_path.mkdir(parents=True, exist_ok=True)
        css_path = assets_path / "report_style.css"

        with open(css_path, "w", encoding="utf-8") as f:
            f.write(self.REPORT_CSS)

        base_args = [
            "--standalone",
            "--embed-resources",
            "--quiet",
            f"--resource-path={md_dir}",
            f"--css={str(css_path)}",
        ]

        # =====================================================================
        # Internal Rendering Functions
        # =====================================================================
        def _render_html(
            src_md: str, out_html: str, is_fallback: bool = False
        ) -> Optional[str]:
            """Internal helper for standard HTML rendering."""
            try:
                pypandoc.convert_file(
                    source_file=src_md,
                    to="html",
                    format="markdown",
                    outputfile=out_html,
                    extra_args=base_args,
                )
                status = "saved" if is_fallback else "generated"
                logger.success(f"HTML report {status}: {out_html}")
                return "HTML"
            except Exception as html_err:
                logger.error(f"HTML conversion failed: {html_err}")
                return None

        def _render_weasyprint(src_md: str, out_pdf: str) -> Optional[str]:
            """Internal helper for WeasyPrint PDF rendering via Pandoc."""
            try:
                logger.info("Attempting PDF export via WeasyPrint...")

                if sys.platform == "win32":
                    f_conf = os.path.join(
                        sys.prefix, "Library", "etc", "fonts", "fonts.conf"
                    )
                    if os.path.exists(f_conf):
                        os.environ["FONTCONFIG_FILE"] = f_conf
                        os.environ["FONTCONFIG_PATH"] = os.path.dirname(f_conf)

                if not self._is_weasyprint_operational():
                    logger.warning("WeasyPrint is unavailable.")
                    return None

                wp_args = base_args + [
                    "--pdf-engine=weasyprint",
                    "--pdf-engine-opt=-q",
                ]

                pypandoc.convert_file(
                    source_file=src_md,
                    to="pdf",
                    format="markdown",
                    outputfile=out_pdf,
                    extra_args=wp_args,
                )
                logger.success(f"PDF generated: {out_pdf}")
                return "WeasyPrint"
            except Exception as e:
                if "permission denied" in str(e).lower():
                    logger.error(f"Permission denied: Close {out_pdf}.")
                    raise e
                logger.warning(f"WeasyPrint engine failed: {e}")
                return None

        def _render_latex(
            src_md: str, out_pdf: str, is_fallback: bool = False
        ) -> Optional[str]:
            """Internal helper for XeLaTeX PDF rendering."""
            try:
                mode = "fallback" if is_fallback else "primary"
                logger.info(f"Attempting PDF export via LaTeX ({mode})...")

                if not self._is_xelatex_available():
                    logger.warning("LaTeX is unavailable.")
                    return None

                if not self._is_rsvg_operational():
                    logger.warning("rsvg-convert is unavailable.")
                    return None

                lx_args = base_args + [
                    "--pdf-engine=xelatex",
                    "-V",
                    "geometry:margin=25mm",
                    "-V",
                    "tables=true",
                ]
                pypandoc.convert_file(
                    source_file=src_md,
                    to="pdf",
                    format="markdown",
                    outputfile=out_pdf,
                    extra_args=lx_args,
                )
                logger.success(f"PDF generated: {out_pdf}")
                return "XeLaTeX"
            except Exception as e:
                if "permission denied" in str(e).lower():
                    logger.error(f"Permission denied: Close {out_pdf}.")
                    raise e
                logger.warning(f"LaTeX engine failed: {e}")
                return None

        # =====================================================================
        # Rendering Route and Guaranteed Cleanup
        # =====================================================================
        overall_success = True

        try:
            for md_path in md_paths:
                md_str = str(md_path)
                file_label = os.path.basename(md_str)
                logger.info(f"--- Exporting PDF for: {file_label} ---")

                base_name = os.path.splitext(md_str)[0]
                pdf_path = base_name + ".pdf"
                html_path = base_name + ".html"

                target = pdf_engine.lower()
                final_engine = None

                # Execute conversion based on target and handle fallbacks
                if target == "html":
                    final_engine = _render_html(md_str, html_path, False)
                elif target == "xelatex":
                    final_engine = _render_latex(
                        md_str, pdf_path
                    ) or _render_html(md_str, html_path, True)
                elif target == "weasyprint":
                    final_engine = (
                        _render_weasyprint(md_str, pdf_path)
                        or _render_latex(md_str, pdf_path, True)
                        or _render_html(md_str, html_path, True)
                    )
                else:
                    logger.error(f"Unsupported engine: {pdf_engine}")
                    overall_success = False
                    continue

                if final_engine:
                    logger.success(
                        f"[{file_label}] completed using {final_engine}."
                    )
                else:
                    logger.error(f"Failed to export [{file_label}].")
                    overall_success = False

            return overall_success

        finally:
            # Phase 2: Cleanup of the runtime CSS file
            if css_path.exists():
                try:
                    css_path.unlink()
                except OSError as e:
                    logger.debug(f"Failed to remove runtime CSS: {e}")
