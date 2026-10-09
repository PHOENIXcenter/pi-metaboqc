# π-MetaboQC: a comprehensive and traceable workflow for automated quality assessment and preprocessing of LC-MS untargeted metabolomics data

[![PyPI version](https://badgen.net/pypi/v/pi-metaboqc)](https://pypi.org/project/pi-metaboqc/)
[![Python 3.10+](https://badgen.net/badge/python/3.10%2B/blue)](https://www.python.org/downloads/)
[![License: MIT](https://badgen.net/badge/license/MIT/blue)](https://github.com/PHOENIXcenter/pi-metaboqc/blob/main/LICENSE)

**π-MetaboQC** is a traceable, adaptive Python workflow for quality control and preprocessing of LC-MS metabolomics feature-intensity matrices. It combines missingness-aware feature filtering, preservation-aware method selection, stage-wise quality assessment, and automated reporting for large, multi-batch studies. The project is distributed on PyPI as `pi-metaboqc` and exposes the Python package `pimqc`.

![Pipeline of π-MetaboQC](https://github.com/PHOENIXcenter/pi-metaboqc/raw/main/docs/pipeline_of_pi-metaboqc.png)

The overview figure is a conceptual workflow summary. In v1.5.0, the framework-neutral object produced after construction is `MetaboDataset`. `MetaboDatasetBuilder.run_build()` returns this dataset and its typed audit through `StageResult`, the same boundary used by later processing actions. The same boundaries are intended to support future workflow-plugin and LLM-skill adapters while keeping those integrations outside the scientific core. The figure should therefore be read together with the current object contracts in the [native API reference](https://github.com/PHOENIXcenter/pi-metaboqc/blob/main/docs/native_api.md).

> [!IMPORTANT]

> **Version 1.5.0 adds an optional R compatibility backend.** The 1.4.0 breaking data-model boundary remains in force: the former pandas-subclass-based `MetaboInt` API has been removed. Native integrations should use `MetaboDataset` and typed `StageResult`/`AuditPayload` contracts. See the [changelog](CHANGELOG.md) for the complete migration boundary.

> **Version 1.4.0 is a breaking data-model release.** The former pandas-subclass-based `MetaboInt` API has been removed. Native integrations should now use `MetaboDataset` and typed `StageResult`/`AuditPayload` contracts. See the [Changelog](https://github.com/PHOENIXcenter/pi-metaboqc/blob/main/CHANGELOG.md) for the complete migration boundary.

## ✨ Core Capabilities

* **Matrix-level LC-MS metabolomics QC workflow:** π-MetaboQC focuses on feature-intensity matrices from large, multi-batch metabolomics studies. It integrates dataset construction, missing-value triage, blank/QC filtering, signal correction, imputation, normalization, QA diagnostics, and report generation in a single reproducible workflow.

* **Explicit Python-native data model and selectable implementations:** The core `MetaboDataset` keeps the intensity matrix, sample metadata, feature metadata, schema, and processing context as separate values. Python implementations remain the default. Users may explicitly select original R calls through the optional `rpy2` backend for VSN, BPCA, QRILC, Metanorm rLOESS, WaveICA 2.0, RUV-III, and a pinned local SERRF source adapter; the selected provider, source identity, package versions, parameters, and transformations are retained in the stage audit.

* **Route-aware filtering and imputation:** Detection patterns, biological groups and QC intensity guide feature retention and handling. The reconstruction route (**R-route**, unrelated to the R programming language) evaluates candidate imputers; the special-handling route (**S-route**) retains features through group-pattern or low-intensity QC rescue and applies the configured lower-tail or small-value strategy. These are operational routes, not statistical missingness mechanisms. Reconstruction candidates are evaluated with an intensity-aware mask and a combination of total and low-intensity NRMSE, distribution fidelity and study-sample structure preservation.

* **Evidence- and preservation-aware adaptive selection:** `AUTO` selects among eligible candidates using predefined stage-specific metrics and study-sample structure preservation. Correction combines QC precision and D-ratio with sample-structure preservation; imputation evaluates masked reconstruction and distribution fidelity; normalization evaluates QC alignment, variance and structure. The selected method, candidate metrics and evaluation basis remain available for audit.

* **Typed and serializable stage contracts:** Each stage returns a `StageResult` containing the data product and a stage-specific `AuditPayload`. Versioned, checksummed serialization makes these boundaries reproducible across Python sessions and suitable for workflow-plugin or LLM-skill adapters without relying on hidden DataFrame state.

* **End-to-end quality assessment and traceability:** A fixed QA suite tracks QC or batch correlation, RSD distributions, PCA structure, and multivariate outliers across stage snapshots. Processing dashboards carry method-selection evidence; QA dashboards show how each processing stage changed the data.

* **Parallel computation and scalable execution:** Computationally intensive steps, including feature-wise correction, model fitting, candidate evaluation, and large-matrix transformations, use `joblib`, `Numba`, and vectorized Scientific Python routines where appropriate. The workflow is designed for clinical-scale cohorts while remaining runnable from notebooks, scripts, and command-line workflows.

* **Automated reporting and publication-ready visualization:** The pipeline records stage-level decisions, retained feature counts, selected methods, evaluation metrics, and diagnostic figures. Users can generate **Brief** or **Comprehensive** reports in Markdown, HTML, or PDF, while diagnostic plots are exported as editable **SVG** or **PDF** files for downstream inspection and manuscript preparation.


## 📦 Installation

We strongly recommend installing π-MetaboQC (`pi-metaboqc`) within a **Conda** virtual environment using [Miniforge](https://github.com/conda-forge/miniforge) (preferred), [Miniconda](https://docs.anaconda.com/free/miniconda/), or [Anaconda](https://www.anaconda.com/download). The package metadata supports Python 3.10 and newer; Python 3.13 is the recommended environment for the current release.

Report conversion requires Pandoc. PDF rendering additionally requires WeasyPrint and its graphical libraries, or an optional XeLaTeX installation with `rsvg-convert` for SVG figures. On Windows, Conda is recommended for provisioning the graphical dependencies.

Follow the installation steps below to prepare the recommended WeasyPrint environment before running report export.

> **Report fallback:** `reporter.export_report(pdf_engine="weasyprint")` tries WeasyPrint first, then XeLaTeX, and finally HTML. Specifying WeasyPrint selects the preferred engine; it does not disable fallback. Missing tools are never downloaded or installed automatically. Install XeLaTeX separately if you need the LaTeX fallback. Pandoc is required for all three routes.

> **Export status:** Successful HTML fallback counts as successful report export, so a `True` return value does not guarantee that a PDF was generated. Check the log and output files for the actual format. Failed report conversion does not remove completed scientific outputs or generated Markdown reports. The CLI exits with `0` on success, `1` on input or processing failure, and `2` when processing completes but report export fails.

### Step 1: Create and Activate Conda Environment

```bash
conda create -n metaboqc python=3.13 pip -y
conda activate metaboqc
```

### Step 2: Pre-install Graphical Engines (Recommended)

Install `pandoc`, `weasyprint`, `tinycss2` and `librsvg` via `conda-forge` to ensure all necessary system graphical libraries are correctly linked before installing the Python package:

```bash
conda install -c conda-forge pandoc weasyprint tinycss2 librsvg -y
```

### Step 3: Install `pi-metaboqc`

**For standard users:**
Install the stable release directly from PyPI:

```bash
pip install pi-metaboqc
```

Alternatively, install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/PHOENIXcenter/pi-metaboqc.git
```

**For developers (Editable mode):**
If you plan to modify the source code or contribute to the project:

```bash
git clone https://github.com/PHOENIXcenter/pi-metaboqc.git
cd pi-metaboqc
pip install -e .
```

### Optional original R implementations

R is **not required** for the default Python workflow. To call supported original R implementations, follow the [optional R backend installation guide](docs/r_backend.md), then select `implementation="r"` in the relevant stage configuration. The guide covers R/rpy2, required packages, the separately obtained SERRF source and dependency checks.

For API examples and input requirements, see [optional R implementations](docs/native_api.md#135-optional-r-implementations). R-enabled `AUTO` selection may include Python candidates where no R adapter exists; failed R calls never silently fall back to Python.

## 🚀 Quickstart & Tutorials

After installing the dependencies, the pipeline accepts three input files: a sample metadata table, a feature-intensity matrix, and a TOML or JSON configuration file. The bundled demo supplies all three and uses `global_seed = 123`, shared with the supplementary case studies and manuscript analyses.

We provide execution modalities for different use cases in the `examples/` directory. **For first-time users, we strongly recommend starting with the Interactive Notebook.**

| Resource | Purpose |
| --- | --- |
| [Interactive Notebook Tutorial](https://github.com/PHOENIXcenter/pi-metaboqc/blob/main/examples/interactive_tutorial.ipynb) | Step-by-step processing and QA dashboards |
| [Headless CLI Example](https://github.com/PHOENIXcenter/pi-metaboqc/blob/main/examples/run_pimqc.py) | Scripted execution for production and batch workflows |
| [Native Python API Reference (1.5.0)](https://github.com/PHOENIXcenter/pi-metaboqc/blob/main/docs/native_api.md) | Dataset and audit contracts, current method signatures, configuration, plotting, serialization, reporting, and optional R backends |

### 1. Interactive Notebook (Recommended for Onboarding)

**[Interactive Tutorial (`interactive_tutorial.ipynb`)](https://github.com/PHOENIXcenter/pi-metaboqc/blob/main/examples/interactive_tutorial.ipynb)**: The maintained, end-to-end introduction to π-MetaboQC. It presents the native API step by step and displays the corresponding processing and QA dashboards at each stage. Use this notebook as the canonical template for adapting the workflow to a new dataset.

Download and open the notebook in VS Code or Jupyter to inspect its saved outputs if GitHub's preview is unavailable. It reads the bundled configuration without imposing extra worker limits; edit the configuration to suit your machine. The tutorial contains standard processing and QA examples, not manuscript-specific exports.

### 2. Headless CLI Execution (For Production & Batch Processing)

For deployment on HPC clusters or integration into larger bioinformatics workflows, utilize our robust command-line interface script (`run_pimqc.py`). 

```bash
# Navigate to the examples directory
cd examples

# Option A: Run out-of-the-box with bundled demo data
python run_pimqc.py

# Option B: Run with your own custom clinical cohort
python run_pimqc.py \
    --meta /path/to/your_meta.csv \
    --intensity /path/to/your_intensity.csv \
    --config /path/to/custom_params.toml \
    --outdir /path/to/output_directory

# Option C: Hide progress and non-error logs (errors remain visible)
python run_pimqc.py -q
```

> ⚠️ **Troubleshooting Note for VS Code Users:** When running the CLI script via the integrated terminal in Visual Studio Code, the IDE may occasionally fail to properly inherit full Conda environment variables. This prevents the PDF rendering engine from locating essential system-level C libraries (e.g., GTK3/Pango), causing the report generation to gracefully degrade and output an **HTML** report instead. 

> **Resolution:** You can bypass this by executing the script from a native system terminal (e.g., Anaconda Prompt, macOS Terminal). Alternatively, to permanently configure VS Code for seamless PDF rendering and resolve PowerShell restrictions, please refer to our **[VS Code Environment & Troubleshooting Guide](https://github.com/PHOENIXcenter/pi-metaboqc/tree/main/docs/vscode_conda_troubleshooting_guide.md)**.

## 📂 Project Structure

```bash
pi-metaboqc/
├── README.md
├── pyproject.toml
├── LICENSE
├── docs/                     # API, optional R setup and environment help
├── examples/
│   ├── interactive_tutorial.ipynb
│   └── run_pimqc.py
├── src/pimqc/
│   ├── core/                  # Core data model
│   ├── config/                # Configuration schema and resolution
│   ├── dataset/               # Dataset construction and validation
│   ├── io/                    # Configuration and filesystem I/O
│   ├── runtime/               # Opt-in logging, progress, and diagnostics
│   ├── processing/            # Processing stages and shared lifecycle
│   │   ├── audit.py           # Typed stage audit payloads
│   │   ├── assessment/
│   │   ├── correction/        # Orchestration and correction engines
│   │   ├── filtering/         # Filtering analysis and execution
│   │   ├── imputation/
│   │   ├── normalization/
│   │   ├── methods.py         # Shared method specifications
│   │   └── stage.py           # Compute/export/render lifecycle
│   ├── statistics/            # Metrics, PCA, and candidate selection
│   ├── plotting/              # Shared plotting infrastructure
│   │   ├── payloads.py        # Immutable plotting contracts
│   │   ├── assessment/
│   │   ├── correction/
│   │   ├── filtering/
│   │   ├── imputation/
│   │   └── normalization/
│   ├── reporting/             # Report assembly and rendering
│   ├── templates/             # Markdown report templates
│   ├── resources/demo/        # Example tables and TOML/JSON configurations
│   ├── serialization/         # Versioned artifact persistence
│   └── pipeline.py            # Automated pipeline orchestrator
└── tests/
    ├── unit/                  # Fast isolated behavior
    ├── integration/           # End-to-end pipeline execution
    ├── quality/               # Source conventions
    └── reference/             # Optional production R-backend checks
```

> **Note on configuration:** Non-`None` stage `run_*` overrides take priority over constructor overrides, followed by the validated TOML/JSON configuration and built-in defaults. The bundled demo intentionally overrides some built-in values. Most users can change datasets and settings without modifying package source code.

## 📖 Supplementary Analyses

Research comparisons, case studies and sensitivity analyses are maintained in [pi-metaboqc-supplementary](https://github.com/PHOENIXcenter/pi-metaboqc-casestudy), independently of the core package. The link retains the currently configured GitHub address; renaming the local checkout does not rename the remote repository.

## 🤝 Contributing & License

This project is licensed under the **MIT License**.
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/PHOENIXcenter/pi-metaboqc/issues).

### AI-assisted development

Codex (OpenAI) assisted with project refactoring and the Python reimplementation of selected R algorithms.

The project maintainers remain responsible for scientific decisions, code review, validation, and releases. AI-assisted implementation does not by itself establish correctness or Python–R equivalence; such claims require the documented comparisons and their limitations. The original algorithms and upstream R code remain credited to their respective authors. AI tools are not credited as authors.
