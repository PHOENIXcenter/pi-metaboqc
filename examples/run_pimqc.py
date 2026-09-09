#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Orchestration CLI for pi-metaboqc Pipeline.

This script provides a standardized command-line interface (CLI) to
execute the full metabolomics quality control workflow in a non-interactive,
headless terminal environment.

Usage:
    python run_pimqc.py [-m META] [-i INTENSITY] [-c CONFIG] [-o OUTDIR] [-q]

    # 1. Run with default demo data
    python run_pimqc.py

    # 2. Run with custom clinical data
    python run_pimqc.py -m /path/to/meta.csv \\
                        -i /path/to/intensity.csv \\
                        -c /path/to/config.toml \\
                        -o /path/to/output_dir
                        
    # 3. Run in silent mode (For background processing)
    python run_pimqc.py -q
"""

import argparse
import os
import sys
from importlib.resources import files

import pandas as pd
from loguru import logger

# =============================================================================
# PRE-FLIGHT CONFIGURATION & BACKEND INITIALIZATION
# -----------------------------------------------------------------------------
# Explicitly set the Matplotlib backend to 'Agg' (Anti-Grain Geometry).
# This is critical for headless server-side execution to prevent GUI
# dependencies and X11 display-related crashes during automated plotting.
# =============================================================================
import matplotlib as mpl

mpl.use("Agg")

# Import internal modules after backend initialization
import pimqc
from pimqc.io import load_pipeline_config
from pimqc.pipeline import run_pipeline


def parse_arguments() -> argparse.Namespace:
    """
    Constructs and parses the command-line arguments for the CLI.

    Returns:
        argparse.Namespace: An object containing parsed arguments.
    """
    # Resolve package data directory and default output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = str(files("pimqc") / "resources" / "demo")
    default_out = os.path.join(script_dir, "tutorial_output")

    parser = argparse.ArgumentParser(
        description=(
            "pi-metaboqc CLI: Automated Metabolomics QC Pipeline. "
            "Optimized for headless deployment and batch processing."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-m",
        "--meta",
        type=str,
        default=os.path.join(data_dir, "project_meta.csv"),
        help="Path to the metadata CSV file.",
    )
    parser.add_argument(
        "-i",
        "--intensity",
        type=str,
        default=os.path.join(data_dir, "project_intensity.csv"),
        help="Path to the intensity matrix CSV file.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=os.path.join(data_dir, "pipeline_parameters.toml"),
        help="Path to the configuration TOML or JSON file.",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default=default_out,
        help="Directory for exporting analytical results and reports.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress progress and non-error console logs.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main execution function orchestrating data ingestion and pipeline flow.
    """
    # 1. Parse command-line inputs before initializing optional runtime
    # diagnostics. Argument defaults point to the packaged demonstration data.
    args = parse_arguments()

    # 2. Configure logging, progress reporting, and hardware checks. Report the
    # installed package version before loading data or running any stage.
    if args.quiet:
        pimqc.init(check_hardware=False, log_level="ERROR", show_progress=False)
    else:
        pimqc.init(check_hardware=True, log_level="INFO", show_progress=True)
    logger.info(f"pimqc.__version__: {pimqc.__version__}")

    # 3. Normalize user-supplied paths.
    # -------------------------------------------------------------------------
    # Absolute paths remove redundant '..' segments, make log messages
    # unambiguous, and keep report assets rooted in one output workspace.
    clean_meta = os.path.abspath(args.meta)
    clean_intensity = os.path.abspath(args.intensity)
    clean_config = os.path.abspath(args.config)
    clean_outdir = os.path.abspath(args.outdir)

    # 4. Log an execution header before any filesystem or data access.
    logger.info("=" * 79)
    logger.info("Starting pi-metaboqc Automated Quality Control Pipeline")
    logger.info("-" * 79)
    logger.info(f">>> Metadata   : {clean_meta}")
    logger.info(f">>> Intensity  : {clean_intensity}")
    logger.info(f">>> Config     : {clean_config}")
    logger.info(f">>> Output Dir : {clean_outdir}")
    logger.info("=" * 79)

    # 6. Load validated configuration and aligned input tables.
    try:
        logger.info("Ingesting configuration and raw data matrices...")

        params = load_pipeline_config(config_path=clean_config)

        # Metadata describes sample identity, roles, batch, and injection order.
        meta_df = pd.read_csv(clean_meta, header=[0])

        # The intensity table uses feature identifiers as its row index.
        int_df = pd.read_csv(clean_intensity, index_col=[0], header=[0])

    except FileNotFoundError as fnf_err:
        logger.error(f"File not found: {fnf_err}")

        sys.exit(1)
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        sys.exit(1)

    # 7. Execute the complete native orchestration path.
    try:
        logger.info("Triggering algorithmic core. Processing...")

        # MetaboDatasetBuilder constructs the raw dataset and typed Audit.
        # The runner then performs independent sample and
        # feature missingness filtering, signal correction, feature-quality
        # filtering, imputation, normalization, QA, and report generation.
        # Its PipelineResult retains the final MetaboDataset, every typed
        # StageResult/Audit, cross-stage QA metrics, and report metadata.

        pipeline_result = run_pipeline(
            meta_df=meta_df,
            int_df=int_df,
            params=params,
            output_dir=clean_outdir,
        )

        if not pipeline_result.report_generated:
            logger.error(
                "Scientific processing completed, but report export failed. "
                f"Stage outputs remain available at {clean_outdir}."
            )
            sys.exit(2)

        logger.success("=" * 79)
        logger.success(
            "Execution successful! Audit reports generated successfully."
        )
        logger.success(
            "Final normalized matrix: "
            f"{pipeline_result.data.intensity.shape[0]} features x "
            f"{pipeline_result.data.intensity.shape[1]} samples."
        )
        logger.success(f"   --> {clean_outdir}")
        logger.success("=" * 79)

    except Exception as e:
        logger.error(f"Pipeline runtime exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Entry point for CLI execution
    main()
