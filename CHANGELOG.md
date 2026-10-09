# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [PEP 440](https://peps.python.org/pep-0440/) for versioning.

The 1.4.0 notes also document the framework-neutral boundaries introduced to make a future workflow-plugin or LLM-skill integration possible without adding Rachis, QIIME 2, or skill-runtime dependencies to the scientific core. These boundaries are intentional integration contracts, not a bundled plugin or skill implementation.

## [1.5.0] - 2026-10-07

> **Optional R compatibility release:** 1.5.0 keeps the Python-native implementation as the default while allowing users to explicitly call selected original R packages through `rpy2`. R is optional and is never used as an implicit fallback.

### Added

- **[R runtime]** Added a lazy, main-thread `rpy2` execution boundary that checks R and package availability, preserves R random-state settings, validates returned matrices, and records JSON-safe provenance. Windows initialization prepares the selected R installation's native library paths without relying on test configuration.
- **[R implementations]** Added explicit original-package paths for `vsn::vsn2`/`vsn::predict`, `pcaMethods::pca`/`completeObs`, and `imputeLCMD::impute.QRILC`. Quantile normalization remains unchanged.
- **[R correction]** Added original `WaveICA2.0::WaveICA_2.0` and `ruv::RUVIII` calls with validated parameter mappings, injection-order restoration, explicit negative controls and technical-repeat design, and recorded scale transformations. Original kernels receive complete non-Blank calculation copies through the documented temporary-input adapter; Blank values are retained unchanged without numerical clipping. These global methods use descriptive full-fit evidence in AUTO, not fabricated OOF predictions.
- **[SERRF]** Added checksum-verified extraction and `rpy2` execution of the author's pinned Shiny-SERRF `serrfR` function using `ranger`. The adapter isolates UI callbacks without running the application, records source identity, upstream RNG behavior and the runtime NA-indexing fix, and requires a separately supplied local source rather than redistributing upstream code without an explicit license.
- **[Metanorm]** Added the separate `Metanorm-rLOESS` method, calling the original exported Metanorm worker with QC-only fitting, per-batch GCV, and log2/exp2 transformations. It is not an alias for QC-RLSC, does not enter AUTO, and reports OOF as unavailable rather than treating full fits as cross-validation.
- **[Configuration]** Added `implementation = "python" | "r"` to correction, imputation, and normalization, with the same constructor/runtime/configuration priority as other parameters. Unsupported R/method combinations fail explicitly.
- **[R AUTO]** Support original R kernels within existing AUTO candidate sets and stage-specific scoring: robust QC-RLSC/SERRF/RUV-III/WaveICA for correction, BPCA/QRILC for imputation, and VSN for normalization. Other candidates retain their declared Python implementation; failed R candidates never fall back to the same Python method. Record requested, candidate and selected backends separately, including actual execution provenance in reports.
- **[SERRF validation]** Add a held-out-QC adapter that refits the original R function per fold, captures its fitted ranger models and training scales, and predicts excluded QCs without using them in fitting or alignment. Preserve the original full-fit call, share deterministic fold assignments with native SERRF, retain failed-fold evidence, and distinguish this pi-metaboqc validation extension from upstream functionality.
- **[Documentation]** Added R/rpy2 installation, environment diagnostics, optional testing instructions, upstream input constraints, and examples based on the locally verified R 4.5.2 and rpy2 3.6.7 environment.
- **[Scale contracts]** Documented stage, adapter, and original-function transformations, including BPCA centering, QRILC's caller-supplied log scale, intrinsic VSN generalized-log output, and RUV-III inverse-domain failures. Comparisons retain invalid-output evidence without introducing clipping, repeated logging, or an implicit fallback.

### Changed

- **[Robust QC-RLSC]** Align the robust correction design with MetaNorm rLOESS using log-scale quadratic symmetric LOESS, continuous GCV span selection, and mean-restoring residual correction. Provide Python and optional R fitting backends with shared finite-sample support rules and fold-local validation; retain standard raw-scale QC-RLSC separately. Exclude Blank and held-out QC values from the relevant reference calculations, record unsupported predictions, and never substitute full-fit values for missing OOF evidence. Historical robust QC-RLSC results do not validate this revised method.
- **[Correction scoring]** Added feature-wise D-ratio diagnostics using QC dispersion relative to the combined QC and biological-sample dispersion. Correction AUTO combines technical precision (median and feature-wise QC-RSD), D-ratio preservation, and sample-structure preservation with 0.35/0.35/0.30 weights when all input metrics are supported. Input-unsupported metrics are omitted consistently across candidates; candidate-specific unavailable metrics contribute zero. OOF evidence is used where available, with method-specific full-model and eligibility rules documented in the native API. The scorecard uses the existing blue palette and wraps the D-ratio legend label.
- **[Correction dashboards]** The AUTO summary now aligns its score, scorecard, and legend row with four equal-width selected-method diagnostics: QC-RSD distribution, feature-wise QC-RSD improvement, canonical D-ratio distribution, and sample-structure change. Explicit-method diagnostics use a compact 2×2 layout. The candidate appendix compares all evaluated methods on shared QC-RSD and D-ratio axes, labeling OOF, full-model, and unavailable results without a separate boxplot brick for every method.
- **[Correction distributions]** Annotate canonical D-ratio medians for baseline and available OOF/full outputs, using one QC/Sample-combined D-ratio per feature and matching the QC-RSD box widths. Stack candidate QC-RSD and D-ratio panels vertically at the standard 60% display width, align method groups and box widths, retain intra/inter stages and method-specific OOF availability, and remove redundant method/stage labels.
- **[Numeric domains]** Apply fixed stage-specific defaults without interactive output-policy choices: correction converts nonpositive/nonfinite results to missing; imputation repairs only finite nonpositive new predictions from same-role, training-only half-minimum positive raw values; normalization preserves valid signed log/glog output. Remove implicit correction floors and raw-value recovery, retain affected-cell diagnostics, and preserve observed imputation values.
- **[Missingness provenance]** Preserve MAR/MNAR feature labels while recording correction-induced missing sample IDs per feature. Route these new cells through reconstruction independently of original MNAR cells, and consume the provenance after imputation; normalization does not depend on mechanism labels.
- **[AUTO evaluation]** Score delivered-policy data independently of Python–R kernel agreement, retain fixed input support and metric weights, isolate failed candidates, and prevent missing diagnostics from increasing the remaining weights. Exclude nonconverged native VSN from AUTO while marking explicit finite VSN output as degraded.
- **[QA scales]** Replace generic exponentiation and epsilon clipping with explicit raw, log2, log2p1 and VSN diagnostic views. Stabilize exponentiation without changing CV, record unavailable support, label the VSN view as a diagnostic rather than an intensity inverse, and keep centered/scaled RSD unavailable.
- **[Audit provenance]** R executions retain the actual function, R/rpy2/package versions, available upstream commit, parameters, seed/RNG kind, warnings, and scale transformations in the corresponding selection audit. Completed artifacts remain readable and plottable without R.
- **[Selection semantics]** Python implementations and AUTO candidate sets remain the default. A requested R method never silently falls back to Python when dependencies, upstream fitting, or output validation fail.
- **[Reporting]** Reports identify original R providers and package versions, reference R-specific VSN dashboard filenames, and avoid attributing native Python VSN parameters to an R fit.

### Fixed

- **[R SERRF seed]** Full-fit and OOF forests now follow the configured `random_state` through a strict runtime AST adaptation of the sole upstream `set.seed(1)` constant; preserve the verified source file and extraction checksums, record upstream/effective seeds separately, and leave tree count, predictors, and other algorithm settings unchanged.
- **[Robust LOESS safety]** Prevent sparse-QC and reweighted singular fits from entering the unsafe compiled symmetric-LOESS path by applying the original bisquare iterations explicitly with shared weighted-design checks. Isolate native fits in a worker process so a compiled-library failure cannot terminate the notebook kernel; retain unavailable-fit evidence and never replace failed correction with raw values or another language backend.
- **[SERRF NA indexing]** Fix upstream logical subscripts containing NA during zero/negative-value repairs through a shared, audited runtime patch for full-fit and OOF calls; preserve the verified source file and original replacement distributions. Exclude and report features with an entirely missing fitting batch instead of fabricating a batch reference, restore their unavailable output positions, and determine fold support without held-out QC information. Retain original NA restoration for downstream imputation and complete-input reference checks.
- **[R missing-value compatibility]** Remove the overly strict complete-positive input guard from original SERRF and pass NaN/zeros unchanged to its upstream stochastic repairs, recording repair ownership and input counts. Preserve missing held-out QC truth and unavailable prediction support in OOF validation. Verify WaveICA/RUV-III's original wavelet/SVD limitations and distinguish them from adapter-based completion.
- **[Temporary missing inputs]** Share feature-median calculation copies across native/R WaveICA and RUV, using observed non-Blank values on the appropriate raw/log scale and restoring original missing positions after correction. Record the extension in logs and audit, reject reference-free features, and require real RUV control support without substituting controls. Keep SERRF's original stochastic fitting repairs distinct from median input preparation.
- **[SERRF missing outputs]** Restore original non-Blank NA positions in R SERRF full-fit and OOF outputs before the common correction-domain policy, preserving missing targets for the later imputation stage. Retain the original R stochastic repairs internally and preserve Blank values; do not additionally convert input zeros to NA. Record the output policy and restored-cell count, and distinguish adapted stage outputs from original-function results in reference comparisons. Restoring NA does not undo temporary repairs' influence on the fitted correction.
- **[QRILC fitting support]** Fit both implementations using all observed input features within each sample, then write back only actual MNAR targets. Avoid degenerate fits caused by restricting training to a few MNAR features, preserve MAR/correction-reconstruction routing, and summarize native fallback warnings through Loguru.
- **[Backend logging]** Mark stage logs with distinct colored `[PYTHON]` and `[R ORIGINAL]` labels while retaining readable labels without color and restoring context after execution.
- **[QRILC]** Corrected the native sampling SD to follow the actual one-dimensional `tmvtnorm` branch used by `imputeLCMD`, removing an unintended square root. Preserve signed log-scale draws, validate tuning parameters, explicitly warn about native constant-fill extensions, and keep kernel evidence separate from the common production repair of finite nonpositive imputations. NaN, infinity and inverse overflow remain failures.
- **[RUV diagnostics]** Native correction accepts explicit controls without empirical augmentation and records effective controls/rank and unconstrained output-domain counts in its audit. Invalid raw inverses become missing at the production correction boundary rather than being floored.
- **[BPCA]** Center native input before zero-fill covariance initialization and reconstruct missing entries from final scores and updated loadings, matching the inspected Bayesian update/stopping rules. Preserve observed values and signed kernel outputs; remove output clipping and reject nonfinite reconstruction.
- **[VSN]** Replace the shared-scale native fit with sample-specific affine calibration, an analytical profile-likelihood gradient, and intensity-stratified robust LTS refinement. Remove the empirical post-fit shift, retain actual sample coefficients and optimizer status in the audit, and update report terminology to describe the fitted model without claiming convergence when it was not achieved.
- **[SERRF]** Rebuild native correction around batch-wise forests, intersected QC/Sample predictor rankings, role-specific standardization and cross-batch alignment. Refit selection, transformations and response scaling inside QC folds, remove the obsolete pooled-correlation input, and retain explicit full/OOF diagnostics for unsupported fits and invalid baselines. Invalid full-fit and held-out QC predictions remain missing rather than being replaced by original intensities, full-fit predictions or unchanged training evidence.

## [1.4.0] - 2026-09-09

> **Breaking architecture release:** 1.4.0 replaces the pandas-subclass-based `MetaboInt` model with an explicit, composition-based data model. It also establishes typed and serializable stage boundaries for native Python, workflow-plugin, and LLM-skill integrations. Code written against the former `MetaboInt` API must migrate to `MetaboDataset` and the new stage contracts.

> **Integration rationale:** Relative to the v1.3.1 release, the v1.4.0 changes deliberately stabilize the data, stage, audit, plot-payload, and serialization boundaries needed by possible Rachis/QIIME 2 plugin and LLM-skill adapters. The adapters remain separate projects and are not required for native `pimqc` execution.

### Added

- **[Dataset actor]** Promoted `MetaboDatasetBuilder` to the recommended root-level construction actor. Its timed `run_build()` method uses the common compute/export/render lifecycle and returns `StageResult[MetaboDataset]` with `DatasetAuditPayload`; `compute_build()` prepares the same data and audit in memory.
- **[Data model]** Added `MetaboDataset`, `DatasetSchema`, `ProcessingContext`, and `SampleRoleLabels` as explicit containers for the intensity matrix, sample metadata, feature metadata, schema, and processing context.
- **[Stage contracts]** Added typed `AuditPayload` implementations for dataset construction, quality assessment, sample missingness filtering, feature missingness filtering, feature-quality filtering, correction, imputation, and normalization. Every action returns its data product and typed audit through `StageResult`.
- **[Independent filtering]** Added `SampleMissingValueFilter`, `FeatureMissingValueFilter`, `FeatureQualityFilter`, and `FilteringOrchestrator` so the three filtering actions can run and be audited independently or be composed by the native pipeline.
- **[Plotting contracts]** Added detached plot-payload snapshots and public audit renderers so plots can be reproduced without a live processor or a recalculation of scientific diagnostics.
- **[QA comparison]** Added audit-driven patchworklib cross-stage QA dashboards for RSD, PCA, QC/batch correlation, and outliers. Each grid has a final independent legend brick, complete statistical outlier categories, data-dependent IS/ORF entries, consistent batch markers, and a half-brick-height correlation colorbar.
- **[Serialization]** Added a versioned, checksummed, non-pickle directory artifact format for `MetaboDataset`, typed `AuditPayload`, and `PlotPayload` roots. Manifests record schema identity, members, sizes, and SHA-256 checksums; strict reads also reject unlisted files.
- **[Native API reference]** Added a v1.4 reference covering public imports, explicit dataset construction, stage configuration, independent filtering, plotting payloads, serialization, reporting, and the migration boundary from v1.3.1.
- **[Integration contracts]** Added framework-neutral object and serialization boundaries so future workflow-plugin and LLM-skill adapters can consume data products, typed audits, and plot-ready diagnostics without reconstructing hidden pandas state or importing execution-only caches.

### Changed

- **[Dataset entry point]** Migrated the native pipeline and tutorial to `MetaboDatasetBuilder.run_build()`, including report metrics from the builder's audit. The pipeline no longer assembles the raw dataset audit itself. `build_dataset()` remains a thin, data-only compatibility wrapper; raw CSV and acquisition-overview filenames are unchanged.
- **[Architecture]** Migrated processors, pipeline orchestration, dataset construction, plotting, reporting, and tests to use ordinary pandas objects inside `MetaboDataset` rather than inherited DataFrame state.
- **[Configuration]** Apply the same validated configuration schema and precedence rules to file loading, direct processor construction, and run-time overrides. Unknown and removed configuration fields now fail at the public boundary.
- **[Diagnostics]** Compute sample-structure coordinates and scores before plotting and retain normalization QC variance/structure diagnostics for processor-free redraws. Final imputation structure diagnostics describe the completed matrix and remain distinct from masked reconstruction benchmarks.
- **[Audit storage]** Store correction candidates/predictions and normalization selection passports once in their plot payloads; audit properties retain convenient read access without duplicate serialization.
- **[Report QA]** Pass an ordered mapping of explicit QA audits to the patchworklib renderer instead of reading and stitching intermediate SVG files. Existing report asset keys and filenames are retained; the legacy SVG compositor remains disabled for reference during the transition.
- **[Reporting]** Build comprehensive and brief narratives from `ReportInput`, validate local image references and Markdown table captions before export, and retain optional HTML/PDF generation as a report-layer concern.
- **[Report layout]** Tuned the print stylesheet for A4 output with stable margins, typography, image aspect-ratio limits, table sizing, repeated table headers, and pagination guards while preserving portable Markdown layout.
- **[Filtering artifacts]** Grouped sample- and feature-level missingness outputs under one Step 02 artifact directory while retaining their independent `StageResult` and typed Audit contracts.
- **[Runtime timing]** Applied execution-time logging consistently to every complete stage lifecycle, including the independent filtering actions and normalization.
- **[Runtime identity]** Report the installed `pi-metaboqc` version at the start of native pipeline, CLI, and interactive-notebook execution, and retain the same version in `ReportInput` metadata.
- **[Tutorial]** Restored the optional IPython autoreload development utility, standardized tutorial filesystem assembly on `os.path.join`, and expanded stage documentation around data products, typed audits, skipped actions, QA interpretation, and plugin/skill hand-off boundaries.

### Fixed

- **[Filtering]** Corrected sample tracking to use schema-defined sample IDs and propagated the executed upstream sample threshold into the combined missingness audit.
- **[Correction]** Preserved floating-point output matrices through skipped imputation, retained AUTO requests across repeated runs, removed training-prediction substitution for unavailable OOF evaluation, adapted QC-RLSC folds to available QC counts, and recorded validation availability, effective folds, coverage, and evaluation basis. AUTO selection records individual candidate failures and continues evaluating viable methods.
- **[Imputation]** Normalized mechanism labels, rejected unknown labels and incomplete target outputs, distinguished MNAR-only execution from MAR benchmarking, and recorded pooled fallback fills. Metrics reflect completed execution even when queried before running.
- **[Normalization]** Separated request configuration from execution metadata so repeated AUTO runs remain AUTO and early metric access cannot freeze pending values. Explicit random seeds propagate to the output dataset context.
- **[Plotting payloads]** Detached nested array snapshots and moved correction QC-RSD and normalization RLE/significance calculations ahead of rendering.
- **[Annotation layout]** Included opaque annotation background padding in candidate-size calculations instead of adding it only to the reported final rectangle; the annotation boundary regression passes in the local Python 3.10 native suite.
- **[Reporting]** Supported array-valued report metadata, removed missing dashboard references and inaccurate MAR/MNAR narratives in degraded reports, and checked export status in CLI and tutorial callers. Retained WeasyPrint, XeLaTeX, and HTML fallback without automatically downloading or installing dependencies.
- **[Reporting]** Removed isolated PDF dashboard headings, kept captions attached to their figures, and verified both comprehensive and brief reports with real WeasyPrint rendering for demo and no-missing datasets.
- **[Pipeline]** Made filtering results explicit dataclass state and included sample filtering in the stage-results mapping.
- **[Configuration]** Validated constructor and runtime settings consistently with file configuration, including method casing, thresholds, worker counts, explicit joblib batch sizes, and numeric SVR gamma.
- **[Documentation]** Synchronized the native API reference and README with the current execution contracts and report dependency/fallback behavior.
- **[Testing]** Repaired source-quality gates and added native regression and isolated installed-wheel checks before release publication. Push, pull-request, and manual validation runs do not publish packages; publishing remains restricted to published Release events.
- **[Testing]** Performed local PDF checks for A4 page bounds, replacement glyphs, caption numbering, image references, and rendered-page visual inspection.
- **[Test discovery]** Restricted default pytest discovery to unit, integration, and quality suites so optional reference configuration does not import R/rpy2 during native testing. Removed the global reference-marker exclusion; cross-language checks remain explicitly selectable with `pytest tests/reference`.
- **[Package validation]** Replaced the hardcoded release-version assertion in the installed-wheel check with a comparison against distribution metadata, avoiding workflow edits for each version increment.
- **[Validation]** Passed all 231 default native tests on Windows with Python 3.10.21, together with Ruff, dependency-consistency, and installed-wheel import/resource checks. The maintainer also confirmed all 10 R/rpy2 reference tests across eight bridge scripts passed in the Windows Python 3.13.13 `metaboqc` environment. These local results do not establish a successful remote CI or PyPI publication.
- **[Assessment context]** Added an explicit assessment context-update boundary and cache invalidation so framework adapters can restore stage and log-scale semantics before scale-sensitive QA is calculated, without resetting configured correlation, scaling, or outlier settings.
- **[Saved diagnostics]** Prevented plot-only redraws from recomputing sample distances, structure-preservation scores, normalization QC diagnostics, or method-selection evidence.
- **[Report assets]** Kept the four cross-stage QA dashboard names stable while avoiding stale or duplicated per-stage SVG composition during final report generation.
- **[Imputation]** Fixed AUTO MAR candidate evaluation crashes when KNNImputer received a feature with all values missing in an isolated QC or sample subset. The implementation now preserves feature dimensions and index alignment across supported scikit-learn versions, restores observed values, and applies a conservative log-space fallback for all-missing features.
- **[QA dashboard fallback]** Reworked the no-actionable-outlier-bar assessment dashboard as a native patchworklib 2 x 2 composition: heatmap/PCA and RSD/outlier bricks are first assembled into two horizontal rows and then stacked vertically without automatic brick stretching; the heatmap body remains square, the RSD plot uses the same effective content width as the heatmap, and the complete outlier scatter legend is mounted in the right-hand lane of its own brick using the PCA-style grouped-legend layout instead of a standalone legend brick.
- **[Extreme-data visualization]** Hid outlier barplots that contain no actionable information; placed fixed-width sample-correlation batch annotation strips flush against the heatmap with tick labels outside the strips and no crossing tick marks; centered short sample and batch labels, wrapped longer labels only at safe separators without splitting words or identifiers, and applied the same renderer-aware policy to inter-batch heatmaps; applied one shared fontsize rule to scientific offset text and logarithmic ticks; resolved Blank/QC axes independently with display margins so zero-valued markers remain fully visible and extreme blank intensities no longer compress the QC range.

### Removed

- **[Legacy data model]** Removed `MetaboInt` and its related compatibility adapters. This release does not provide a backward-compatibility wrapper for the former DataFrame-subclass API.
- **[Implicit state]** Removed cross-stage contracts based on `DataFrame.attrs`, shared mutable processor statistics, and other hidden in-memory state.
- **[Legacy result contracts]** Removed generic `StageResult` metrics, candidates, metadata, and audit-table dictionaries together with `LegacyAuditPayload`. Consumers must use each stage's typed audit fields through `result.audit`.

### Breaking Changes

- Public processing inputs and outputs now use `MetaboDataset` and typed `StageResult`/`AuditPayload` contracts.
- Custom plotting code must construct or consume the corresponding plot payload instead of reading mutable processor internals.
- `StageResult` is an in-memory composition boundary rather than a serialized artifact root. Persist `result.data` and `result.audit`; the audit already owns its plot payload.
- Serialized 1.4 dataset, Audit, and plot-payload artifacts use the new versioned directory format and are not interchangeable with ad hoc or pickled files from earlier releases.
- Removed and misspelled configuration fields are rejected instead of being translated or ignored.

## [1.3.1] - 2026-08-19

### Fixed

- **[Reports]** Fixed dashboard placement and ordering in the comprehensive report.
- **[Reports]** Fixed imputation reports incorrectly displaying Jensen-Shannon distance and normalized Wasserstein distance as `N/A`.

## [1.3.0] - 2026-08-18

> **Breaking release:** 1.3.0 refactors the processing-stage and report execution lifecycle. Processing computation, artifact export, rendering, metrics aggregation, and report generation now use explicit structured contracts. Dataset construction retains the `build_dataset(output_dir=...)` execution entry point, while its drawing implementation is separately packaged under `pimqc.plotting`. Existing custom code built against pre-1.3.0 stage or visualization APIs may require migration.

### Added

- **[Stage lifecycle]** Added explicit `compute`, `export`, and `render` execution phases through `StageRunner` and `StageResult`. `run_pipeline()` now returns a structured `PipelineResult`.
- **[Reporting]** Added the `ReportInput` report contract and a persistent `Report_Input.json` snapshot containing the metrics, resolved configuration, metadata, and rendered-asset manifest used to generate the report.
- **[Plotting]** Added the public `pimqc.plotting` namespace for dataset, assessment, filtering, correction, imputation, and normalization visualizations, including dashboards and scorecards.
- **[Configuration]** Added equivalent TOML and JSON configuration loading with the documented precedence of notebook/runtime overrides, pipeline configuration, and module defaults.

### Changed

- **[Public execution]** Updated the pipeline, CLI, notebook, demo resources, and report templates to use the structured stage and report contracts. `build_dataset(output_dir=...)` remains the dataset-construction entry point.
- **[Metrics and reporting]** Pipeline and report generation now consume metrics and audit tables captured during each stage instead of deriving them again from final DataFrames.
- **[Visualization]** Standardized report-facing annotation, reference-line, legend, colorbar, and axis-layout behavior across stage dashboards. Dataset construction and all processing-stage plotting are exposed through the plotting namespace.
- **[Runtime]** Optional R tooling is loaded only when an R-backed method is selected; unavailable R packages are reported as clear skips instead of preventing normal package import.

### Breaking Changes

- Stage methods now return `StageResult` objects, and pipeline execution returns `PipelineResult`.
- Report generation now consumes `ReportInput` rather than reconstructing report data from final DataFrames.
- Former `pimqc.visualization` and stage-local visualization import paths are removed without compatibility wrappers.
- Custom code calling pre-1.3.0 stage or visualization APIs must migrate to the new contracts.

## [1.2.0] - 2026-08-08

### Added

- **[Correction]** Extended QC-RLSC with optional Tukey-bisquare residual reweighting, quadratic LOESS, and QC-only constrained-grid/OOF span selection, while retaining fixed-span linear LOESS as the default. The robust fitting design was informed by the [UGent-LIMET Metanorm project](https://github.com/UGent-LIMET/Metanorm). Blank-aware fitting and frozen-model prediction are applied where the correction method supports them.
- **[Configuration]** Added validated configuration fields for the new QC-RLSC options, including span selection, polynomial degree, minimum QC count, and robust-reweighting iterations.
- **[Visualization]** Added compact, article-ready dashboard builders and dedicated panel/legend layouts for filtering, correction, imputation, and normalization results.

### Changed

- **[Architecture]** Reorganized the former monolithic stage scripts into domain subpackages under `processing/`, separating analysis, visualization, and algorithm responsibilities. `MetaboInt` now lives in `core`, while dataset construction, statistics, visualization infrastructure, reporting, I/O, and configuration each have dedicated packages.
- **[Configuration and packaging]** Added centralized stage-configuration resolution with one precedence rule for defaults, TOML/JSON sections, and runtime overrides. Bundled demo tables/configuration now live under `resources/demo`, and report templates are packaged under `templates`.
- **[Correction]** Consolidated correction candidate labels, parameters, scoring, and diagnostics. AUTO correction compares fixed-span standard and robust QC-RLSC together with the supported alternatives; QC-RFSC remains available for explicit selection, while WaveICA 2.0 uses full-data QC-RSD evaluation and other candidates use OOF evaluation where applicable.
- **[Visualization and reporting]** Centralized adaptive figure sizing, annotation/tick scaling, colorbar and legend styling, and publication-ready SVG layout. Sequential QA dashboards now emit compact shared sidecars, and intermediate source panels/sidecars are removed after final grid assembly.
- **[Documentation]** Updated the runnable notebook, CLI defaults, pipeline descriptions, and project structure documentation for the reorganized seven-stage workflow.

### Fixed

- **[Configuration and resources]** Fixed package-relative resolution of demo data and report templates, added JSON loading alongside TOML, and corrected unsupported-format errors to advertise the formats that are actually supported.
- **[Correction and diagnostics]** Aligned the new QC-RLSC options with AUTO candidate evaluation, method-specific scoring, and downstream dashboards; renamed the span parameter from `loess_frac` to `loess_span` and documented fixed-span and GCV modes consistently.
- **[Reports and visualizers]** Fixed report assembly edge cases involving conditional IS/ORF legends, shared diagnostic sidecars, stale PDF intermediates, and stage-specific dashboard titles.

## [1.1.5] - 2026-07-23

### Added

- **[Shared Metrics]** Added reusable metrics for relative technical improvement, masked-value distribution fidelity, and study-sample structure preservation, enabling consistent `AUTO` evaluation across correction, imputation, and normalization.
- **[Correction]** Added preservation-aware `AUTO` selection that combines median and feature-wise QC-RSD improvements with study-sample structure retention, together with candidate QC-RSD comparison outputs.
- **[Imputation]** Added GMM- and low-intensity-noise masking for MAR candidate evaluation and preservation-aware scoring based on total/low-intensity reconstruction error, masked-value distribution fidelity, and study-sample structure retention; QRILC is included in candidate comparison and reporting.
- **[Normalization]** Added a multi-criterion `AUTO` normalization evaluation using QC RLE alignment, QC variance stabilization, QC structure distance change, and sample structure preservation, with `ROBUST_LOG_ONLY` retained as the neutral reference strategy.

### Changed

- **[Filtering]** Consolidated high-missing-value feature routing around biological-group MNAR rescue, QC-level MNAR rescue, MAR eligibility checking, and explicit exclusion, while applying QC-RSD reproducibility filtering only to eligible MAR features.
- **[Correction/Imputation/Normalization]** Reorganized `AUTO` dashboards around score summaries, candidate preservation scorecards, and selected-method diagnostics; fixed-method workflows now show only the relevant downstream diagnostics.
- **[Correction/Imputation/Normalization]** Aligned score terminology, candidate ranking, and selected-method diagnostics across the three adaptive modules, including score-aligned sample-structure change maps.
- **[Visualizer]** Standardized editable SVG/PDF output across all processing and QA visualizers, including solid equivalents for transparent categorical colors, AI-compatible text handling, adaptive heatmap/tick formatting, and grouped standalone legends.
- **[Visualizer]** Migrated the shared QC/preprocessing visualization accent from `tab:red` to `tab:blue`, including equivalent solid shades used in vector exports. Dataset-construction sample-type plots retain their established red/blue/gray encoding.
- **[Reports]** Updated brief and comprehensive report templates for revised `AUTO` metrics, candidate-comparison figures, fixed-method outputs, skipped-imputation states, and consistent scientific terminology.

### Fixed

- **[Reports]** Fixed Markdown/PDF rendering in correction, filtering, imputation, and normalization sections, including headings, abbreviation reuse, conditional OOF descriptions, and associated figure captions.
- **[Visualizer]** Fixed SVG/PDF export issues affecting heatmap colorbars and cell boundaries, patch backgrounds, alpha rendering, text editability, legend placement, and sequential Patchworklib dashboard export.
- **[Assessment/Filtering]** Fixed dense heatmap, tick-label, legend, PCA annotation, missing-value rescue-region, and filtering-summary rendering edge cases for report-ready figures.

## [1.1.4] - 2026-06-24

### Added

- **[Correction]** Added **WaveICA2** as a configurable correction method and included it in `AUTO` correction evaluation, with R-reference comparison support for WaveICA2, RUV-III, and SERRF.
- **[Imputation]** Added **BPCA** as a MAR imputation candidate in the mask-based NRMSE selection workflow, with configurable convergence settings and R-reference comparison support for BPCA and QRILC.
- **[Normalization]** Added `AUTO` normalization evaluation based on QC RLE alignment change, QC variance stabilization, QC structure distance improvement, and sample structure preservation, using `ROBUST_LOG_ONLY` as the neutral reference strategy.

### Changed

- **[Filtering]** Refined the missing-value classification dashboard and flowchart, including boundary-anchored arrows, clearer MAR eligibility labeling, dynamic QC intensity percentile labels, and consistent node typography for BioGroup and non-BioGroup layouts.
- **[Correction]** Updated correction dashboards, `AUTO` scorecards, method labels, and internal-standard visualizations to accommodate WaveICA2 while retaining the existing QC-RLSC, QC-RFSC, QC-SVR, SERRF, and RUV-III workflows.
- **[Imputation]** Updated MAR candidate reporting and dashboard logic to include BPCA and to bypass imputation cleanly when no missing values require filling.
- **[Normalization]** Refactored the normalization dashboard and score summary around the final `AUTO` score components, including the integrated score-contribution panel, QC RLE alignment plot, QC variance-stabilization trend, QC structure distance plot, and sample structure preservation plot.
- **[Visualizer]** Consolidated legend layout, dense tick-label handling, and shared formatting utilities across dataset-building, filtering, assessment, correction, imputation, and normalization figures.
- **[Reports]** Updated brief and comprehensive report templates to reflect skipped imputation states, the final auto-normalization scoring criteria, and auto-normalization score visualizations without showing irrelevant candidate plots.
- **[Tests]** Split R bridge validation into method-specific scripts for QRILC, BPCA, VSN, quantile normalization, WaveICA2, RUV-III, and SERRF, with R-related temporary files excluded from version control.
- **[Code Quality]** Continued type-hint, formatting, and configuration cleanup across source and test modules using `ruff`, `black`, and `pyproject.toml` metadata updates.

### Fixed

- **[Dataset Builder]** Fixed acquisition-overview legend clipping and sublegend overlap for datasets with many batches or sample types.
- **[Assessment]** Fixed dense tick-label overlap and cell-annotation scaling in QC correlation heatmaps, batch-correlation heatmaps, and integrated outlier bar plots.
- **[Filtering]** Fixed flowchart arrow rendering so connectors attach to node borders rather than relying on fragile shrink offsets.
- **[Imputation]** Fixed no-missing-value datasets so imputation is reported as not required and unnecessary imputation dashboards are omitted.
- **[Normalization]** Fixed normalization score/rendering consistency by using deterministic QC variance-stabilization and structure metrics, centered improvement scoring against the `ROBUST_LOG_ONLY` baseline, and a composite sample structure term based on trustworthiness, distance-rank preservation, and distance-scale preservation.
- **[Reports]** Fixed low-quality filtering feature totals and Section 1.4 paragraph rendering in brief and comprehensive reports.


## [1.1.3] - 2026-06-08

### Changed

- Improved module-level script purpose comments across source and test files.
- Added and refined type hints across the codebase.
- Standardized code formatting with Ruff and Black.
- Cleaned up project metadata and packaging-related configuration.
- Kept package metadata centralized in `pyproject.toml`.

### Validation

- Ruff checks passed for source and test files.
- Black formatting checks passed.
- Python compile checks passed.

### Notes

This release focused on codebase standardization, maintainability, and packaging readiness. It did not introduce user-facing workflow changes relative to `v1.1.2a1`.


## [1.1.2a1] - 2026-06-01

### Added

- **[Dataset Builder]** Added critical feature detection during dataset construction to output prompts for degraded analysis modes.
- **[Dataset Builder]** Added automatic replacement of exact zero values to prevent `ZeroDivisionError` during downstream analytical and normalization steps.
- **[Imputation]** Implemented **LLS** (Local Least Squares) for missing value imputation.
- **[Assessment]** Introduced a unified 3-mode clustering routing (`cluster="total"`, `"within-group"`, `"none"`) for QC correlation heatmaps. The `"within-group"` mode dynamically generates isolated batch forests to reveal intra-batch sub-drifts without disrupting the overarching temporal batch sequence.
- **[IO Utils]** Implemented a custom `joblib` context manager (`tqdm_joblib_env`) to seamlessly integrate `tqdm` with parallel processing backends.

### Changed
- **[Visualizer]** Optimized the marker style display mode; numeric characters are now used as marker styles when the number of groups/batches exceeds 10 to robustly handle extremely large datasets.
- **[Visualizer]** Improved VS Code compatibility by reverting PNG inline rendering to native `IPython.display.Image` mode, enabling native image toolbar (copy/save) support.  Refactored visualization engine with a three-tier configuration strategy (Method > Instance > Global) to independently control `save_format`(["pdf", "svg"] or "svg") and `display_format` ("svg" or "png").
- **[Dataset Builder]** Optimized the `Global_Acquisition_Overview` plot to automatically omit the missing value distribution subplot when no missing values are present, and enhanced the legend display format.
- **[Filter]** Optimized the display of the workflow flowchart and retained feature bar charts during degraded analysis when Blank or Bio Group metadata is missing.
- **[Correction]** Overhauled the display logic of the internal standard (IS) scatter plots across multiple correction stages for improved memory efficiency.
- **[Imputation]** Renamed the **Probabilistic** method to **MinProb** to align with widely adopted academic terminology.
- **[Imputation]** Upgraded the MAR simulation algorithm (`generate_gmm_noise_mask`). Shifted from a global GMM evaluation to a strict **batch-wise independent GMM** masking strategy, accurately capturing the localized Limit of Detection (LOD) and noise baseline fluctuations for each analytical batch. Included a robust fallback mechanism for singleton batches or batch-free datasets.
- **[Imputation]** Optimized the subplot grid layout of `Imputer_Candidates.svg` to dynamically adapt to the number of evaluated algorithms.
- **[Assessment]** Refined the rendering logic of QC correlation heatmaps (dynamic toggling of correlation values and adaptive font scaling) and outlier bar plots (automatic tick font size adjustment and tick skipping when the number of outlier bars is excessively large).
- **[Assessment]** Completely refactored the QC correlation heatmap architecture (`plot_qc_corr_heatmap`). Transitioned from Seaborn's default `clustermap` to a pure Matplotlib `GridSpec` engine, achieving pixel-perfect mathematical alignment between independent lower-triangle dendrograms and heatmap cells.

### Fixed
- **[Correction]** Fixed the "progress bar illusion" during parallel processing (e.g., QC-RFSC and SERRF corrections). The progress bar now accurately tracks actual task completion from background workers rather than just task dispatching.
- **[Visualizer]** Fixed canvas memory leakage in Patchworklib workflows by optimizing the execution sequence (Jupyter rendering prior to physical export).

## [1.1.0a1] - 2026-05-26

### Added
- Added `CHANGELOG.md` to the project root directory.
- **[Correction]** Implemented **SERRF** (Systematic Error Removal using Random Forest) method for signal correction.
- **[Correction]** Implemented **RUV-III** (Remove Unwanted Variation) method, which utilizes Singular Value Decomposition (SVD) to eliminate the need for cross-validation.
- **[Correction]** Introduced **AUTO** mode, which dynamically evaluates multiple correction algorithms and automatically selects the optimal method based on out-of-fold QC RSD.
- **[Imputation]** Added **QRILC** (Quantile Regression Imputation of Left-Censored data) algorithm, specifically tailored for the imputation of MNAR (Missing Not At Random) features.
- **[Normalization]** Added **MDFC** (Median Difference from Control) approach for data normalization.

### Changed
- **[Correction]** Refactored `QC-RLSC`, `QC-RFSC`, and `QC-SVR` algorithms into a single, unified `RegressionCorrector` class for better maintainability.
- **[Correction]** Integrated a SERRF-like cross-validation strategy into the core regression correction pipeline to enhance stability.
- **[Imputation]** Optimized the visualization logic for NRMSE scatter plots and KDE curves.
- **[Normalization]** Accelerated the parameter calculation speed by introducing a downsampling mechanism.

### Fixed
- Fixed specific compatibility issues that caused execution errors on Windows OS and Python 3.13.
- **[Report]** Resolved a rendering issue in `AUTO` mode where reports failed to display the correct algorithm parameters and specific RSD improvement plots.

## [1.0.0a1] - 2026-05-19

### Added
- Initial alpha release of the `pi-metaboqc` package.
