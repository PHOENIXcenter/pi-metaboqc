# 1 Native Python API — pi-metaboqc 1.5.0

This reference describes the composition-based Python API implemented in pi-metaboqc 1.5.0. For a guided workflow with stage-by-stage dashboards, start with the [interactive tutorial](../examples/interactive_tutorial.ipynb). Configuration examples are available in the packaged [TOML](../src/pimqc/resources/demo/pipeline_parameters.toml) and [JSON](../src/pimqc/resources/demo/pipeline_parameters.json) files.

Contents:

- [1.1 Workflow and module responsibilities](#11-workflow-and-module-responsibilities)
- [1.2 Dataset API](#12-dataset-api)
- [1.3 Configuration API](#13-configuration-api)
- [1.4 Processing API](#14-processing-api)
- [1.5 Result and audit API](#15-result-and-audit-api)
- [1.6 Pipeline API](#16-pipeline-api)
- [1.7 Plotting API](#17-plotting-api)
- [1.8 Serialization API](#18-serialization-api)
- [1.9 Reporting API](#19-reporting-api)
- [1.10 Integration boundaries](#110-integration-boundaries)

## 1.1 Workflow and module responsibilities

### 1.1.1 Scientific workflow

The standard pipeline performs dataset construction → sample missingness filtering → feature missingness classification/filtering → signal correction → feature quality filtering → imputation → normalization. Quality assessment (QA) observes the raw dataset and subsequent feature-processing checkpoints, including every selected correction stage. QA does not replace the intensity matrix.

Sample and feature missingness are separate execution boundaries with independent audits. They share the user-facing `02_MV_Filtered` directory and one combined missingness dashboard. Feature quality filtering remains a separate stage after correction; it uses the original feature missingness tracking to preserve R-route/S-route assignments and retention history. R-route means reconstruction route, not the R programming language. S-route means special-handling route for features retained through group-pattern or low-intensity QC rescue. These operational categories do not identify statistical missingness mechanisms.

```text
MetaboDataset
  intensity + sample_metadata + feature_metadata + schema + context
       |
       v
processing stage -> StageResult
                     |-- data: dataset, diagnostic object, or dataset mapping
                     `-- audit: typed AuditPayload
                                  |-- metrics / decisions / tracking
                                  `-- plot_payload: detached rendering inputs
```

Scientific calculations belong to processing/statistics modules. Plotters consume recorded payloads; reports consume recorded metrics, stage figures, and audit-backed comparison figures. Artifact serialization persists datasets, audits, or plot payloads independently of the live processor.

### 1.1.2 Import surfaces

| Module | Responsibility and principal exports |
| --- | --- |
| `pimqc` | Dataset types, builder, processing classes, `StageResult`, `PipelineResult`, `run_pipeline`, `__version__` |
| `pimqc.core` | `MetaboDataset`, `DatasetSchema`, `ProcessingContext`, `SampleRoleLabels`, `DatasetProcessor` |
| `pimqc.dataset` | `MetaboDatasetBuilder`, `build_dataset` |
| `pimqc.config` | `PipelineConfig`, section models, `validate_pipeline_params`, `resolve_stage_config` |
| `pimqc.io` | `load_pipeline_config`, `ensure_directory`, `dir_tree` |
| `pimqc.processing` | `StageResult`, typed audits, processing classes and stage utilities |
| `pimqc.processing.assessment` | `QualityAssessor`, `AssessmentDiagnostics` |
| `pimqc.plotting` | Payload types, `BasePlotter`, snapshot helpers |
| `pimqc.serialization` | Typed artifact readers/writers, manifests, validators and serialization errors |
| `pimqc.reporting` | `ReportInput`, `VisualAssetReporter`, `NarrativeStatsReporter` |

Stage-specific plotters are imported from `pimqc.plotting.assessment`, `filtering`, `correction`, `imputation`, and `normalization`; `DatasetPlotter` is in `pimqc.plotting.dataset`. Import `SampleFilterAuditPayload` from `pimqc.processing`, not the package root.

### 1.1.3 Runtime initialization

```python
import pimqc

pimqc.init(
    check_hardware=False,
    log_level="INFO",
    show_progress=True,
    preserve_existing_sinks=False,
)
```

`init()` explicitly configures the application runtime, including logging and progress behavior, with optional hardware diagnostics. It is not a scientific processing stage. Its defaults are `check_hardware=True`, `log_level="DEBUG"`, `show_progress=True`, and `preserve_existing_sinks=False`.

## 1.2 Dataset API

### 1.2.1 `MetaboDataset`

`MetaboDataset(intensity, sample_metadata, feature_metadata, schema=DatasetSchema(), context=ProcessingContext())` stores ordinary pandas tables; it is not a `DataFrame` subclass. Default schema/context values are independently constructed for each instance.

| Member | Contract |
| --- | --- |
| `intensity` | Feature-by-sample `DataFrame`; flat, unique, non-missing identifiers on both axes |
| `sample_metadata` | `DataFrame` indexed by the same sample IDs as `intensity.columns` |
| `feature_metadata` | `DataFrame` indexed by the same feature IDs as `intensity.index` |
| `schema` | Identifier and metadata semantics, including sample-role labels |
| `context` | Acquisition mode, reference features, random seed, stage label, and explicit log/scale state |

Construction defensively copies the tables and aligns metadata order to the intensity axes. Missing or extra metadata IDs, duplicate IDs, MultiIndex identifiers, and missing required metadata fields raise errors. The stored tables remain separate; an annotated calculation view is not the canonical data representation.

`MetaboDataset.from_tables(intensity, sample_metadata, feature_metadata=None, *, schema=None, context=None, sample_id_column=None, feature_id_column=None) -> MetaboDataset` accepts already prepared tables. If feature metadata is omitted, an empty table with the correct feature index is created. Explicit identifier-column arguments must match the corresponding schema names.

```python
from pimqc import DatasetSchema, MetaboDataset

# intensity is already feature-by-sample; metadata includes Sample Name.
dataset = MetaboDataset.from_tables(
    intensity=intensity,
    sample_metadata=metadata,
    sample_id_column="Sample Name",
    schema=DatasetSchema(biological_group=None),
)
```

This example assumes `metadata` contains `Sample Name`, `Sample Type`, `Batch`, and `Inject Order`. Use the builder below for raw-input cleanup and configured injection-order handling; `from_tables()` does not perform zero-to-missing conversion or duplicate-feature resolution.

### 1.2.2 Schema and context

| Type | Fields and defaults |
| --- | --- |
| `SampleRoleLabels` | `actual="Sample"`, `blank="Blank"`, `qc="QC"`, `additional={}` |
| `DatasetSchema` | `feature_id="Metabolite"`, `sample_id="Sample Name"`, `sample_type="Sample Type"`, `batch="Batch"`, `injection_order="Inject Order"`, `biological_group="Bio Group"`, `roles`, `sample_metadata_order=()` |
| `ProcessingContext` | `acquisition_mode="ESI+"`, `internal_standards=()`, `outlier_reference_features=()`, `global_seed=DEFAULT_RANDOM_SEED`, `pipeline_stage="Raw data"`, `is_logged=False`, `log_base="None"`, `is_scaled=False`, `scale_method="None"`, `extra_attrs={}` |

These are frozen dataclasses; collection defaults use factories. Use `dataclasses.replace()` or `dataset.with_intensity(..., context_updates=...)` to construct updated context. `extra_attrs` is for framework-neutral acquisition annotations, not a substitute for typed scientific audits.

`SampleRoleLabels.from_mapping(sample_dict)` and `.to_mapping()` translate the configuration representation. Its standard keys are `"Actual sample"`, `"Blank sample"`, and `"QC sample"`; their values are the labels found in your metadata. For example, `{"Actual sample": "Study", "Blank sample": "Blank", "QC sample": "Pool"}` identifies metadata values `Study`, `Blank`, and `Pool`.

### 1.2.3 Dataset methods and properties

| Member | Return value or behavior |
| --- | --- |
| `copy()` | Detached `MetaboDataset` copy |
| `with_intensity(intensity, *, context=None, context_updates=None, feature_metadata=None)` | New dataset aligned to replacement intensity, preserving or explicitly updating context and feature annotations |
| `to_tables()` | Tuple of detached intensity, sample-metadata, and feature-metadata tables, in that order |
| `annotated_frame()` | Temporary ordinary `DataFrame` with metadata MultiIndex columns and a settings view in `attrs`; not a cross-stage data contract |
| `settings()` | Schema/context settings dictionary used by calculations |
| `sample_ids`, `feature_ids` | Copies of ordered identifier indexes |
| `is_multi_batch`, `ordered_batches` | Batch summaries; batches ordered by their earliest injection |
| `role_sample_ids(role)` | Sample IDs for `"actual"`, `"blank"`, or `"qc"` |
| `qc_data`, `blank_data`, `actual_data` | Role-specific intensity tables |
| `valid_internal_standards`, `valid_outlier_reference_features` | Configured reference-feature IDs present in the data |
| `intensity_order_info(feature_type="IS")` | Reference intensities and injection-order information; `"ORF"` selects outlier references |
| `dataset_metrics` | Dataset composition, sample-role counts, batch distribution, and package-version metrics |

`with_intensity()` can retain subsets of existing samples/features; adding new IDs requires suitable metadata and usually a fresh `from_tables()` call. Treat stage inputs as immutable values even though the constituent pandas tables are mutable.

### 1.2.4 `MetaboDatasetBuilder` and `build_dataset`

`MetaboDatasetBuilder(meta_info, int_df, pipeline_params=None)` copies raw metadata/intensity tables and validates configuration. Metadata includes the configured sample-name column; intensity has feature IDs as its index and sample IDs as columns.

| Entry point | Return type | Side effects |
| --- | --- | --- |
| `builder.compute_build()` | `StageResult[MetaboDataset]` with `DatasetAuditPayload` | No exports or rendering |
| `builder.run_build(output_dir=None)` | Same result/audit contract | Optional CSV and acquisition dashboard |
| `build_dataset(meta_info, int_df, pipeline_params=None, output_dir=None)` | `MetaboDataset` | Functional wrapper around `run_build(...).data`, including optional exports/rendering |

The builder resolves duplicate features, checks sample/metadata consistency and required fields, handles injection order according to configuration, aligns the matrix, and converts explicit zero intensities to missing values. Prefer `run_build()` when you need the audit as well as the data.

```python
from pimqc import MetaboDatasetBuilder

builder = MetaboDatasetBuilder(
    meta_info=metadata,
    int_df=intensity,
    pipeline_params=params,
)
raw_result = builder.run_build(output_dir="outputs/01_Raw_Data")
raw_dataset = raw_result.data
```

With an output directory, the main artifacts are `Raw_Data_Intensity.csv` and `Global_Acquisition_Overview.svg`. Builder settings come from `pipeline_params["Dataset"]`; unlike processing constructors, the builder does not accept arbitrary dataset-field keyword overrides.

## 1.3 Configuration API

### 1.3.1 Loading and validating

```python
from pimqc.io import load_pipeline_config
from pimqc.config import validate_pipeline_params

params = load_pipeline_config("pipeline_parameters.toml")
validated_params = validate_pipeline_params(params)
```

`load_pipeline_config(config_path)` reads TOML or JSON and returns a normalized, validated dictionary. `validate_pipeline_params(pipeline_params)` applies the same strict section/field schema to in-memory mappings; `None` returns an empty dictionary. Unknown sections and fields are rejected rather than silently replaced with defaults.

`PipelineConfig` and its section models expose Pydantic validation through `model_validate()` and normalized dictionaries through `model_dump()`. The six accepted sections are `Dataset`, `QualityAssessor`, `FeatureFilter`, `SignalCorrector`, `MissingValueImputer`, and `DataNormalizer`.

### 1.3.2 Parameter resolution

The effective priority is:

1. Non-`None` keyword overrides passed to the stage's `run_*` method.
2. Non-`None` constructor overrides.
3. The validated configuration section.
4. Built-in defaults.

`resolve_stage_config(pipeline_params, section_name, defaults, explicit_overrides=None)` merges settings without modifying input mappings and validates the resulting stage values against the Pydantic schema. Run-time overrides also undergo an allowed-key check before computation; accepted changes update the processor configuration and invalidate cached calculations.

Omitting a method means “use the resolved configuration,” not “force Auto.” A configured `norm_method="VSN"` remains VSN if the constructor omits `norm_method`; explicitly passing `norm_method="Auto"` requests selection. Omission and explicit Auto coincide when the resolved method is Auto, including the normal default configuration. Inspect the returned audit to determine what ran.

Constructor and runtime overrides are validated against the same stage schema as file configuration before execution or artifact creation. Case-insensitive method choices, including `AUTO`, are normalized consistently. Invalid thresholds, zero workers, nonpositive explicit joblib batch sizes, and nonpositive numeric SVR gamma raise an error. Direct `global_seed` overrides propagate to the returned dataset context and plot payload.

The demo configuration is an example, not a table of immutable class defaults. Some example tuning values differ from constructor defaults; use the loaded configuration and recorded audit rather than copying presumed defaults into a report.

### 1.3.3 Accepted fields

| Section | Fields |
| --- | --- |
| `Dataset` | `mode`, `sample_name`, `sample_type`, `bio_group`, `group_order`, `batch`, `inject_order`, `boundary`, `global_seed`, `internal_standard`, `outlier_ref_feat`, `resort_inject_order`, `sample_dict` |
| `QualityAssessor` | `corr_method`, `scaling_method`, `is_outlier_threshold`, `orf_outlier_threshold` |
| `FeatureFilter` | `sample_mv_tol`, `mv_global_tol`, `mv_qc_tol`, `mv_group_tol`, `mnar_group_mv_tol`, `mnar_qc_mv_tol`, `mnar_intensity_pct`, `blank_qc_ratio_tol`, `qc_rsd_tol` |
| `SignalCorrector` | `implementation`, `base_est`, `loess_span`, `loess_degree`, `rlsc_span_selection`, `rlsc_span_grid`, `rlsc_min_qc`, `rlsc_robust`, `rloess_span_selection`, `rloess_span`, `rloess_iterations`, `rf_n_tree`, `serrf_n_tree`, `serrf_corr_features`, `serrf_backend`, `serrf_batch_size`, `serrf_r_source`, `svr_kernel`, `svr_c`, `svr_gamma`, `ruv_k`, `ruv_control_features`, `ruv_replicate_column`, `waveica_components`, `waveica_cutoff`, `waveica_alpha`, `waveica_levels`, `waveica_spline_knots`, `waveica_max_iter`, `regression_backend`, `regression_batch_size`, `cv_folds`, `n_jobs` |
| `MissingValueImputer` | `implementation`, `mnar_method`, `mnar_fraction`, `mar_method`, `knn_neighbors`, `lls_neighbors`, `bpca_components`, `bpca_max_iter`, `bpca_tol`, `sim_mask_ratio` |
| `DataNormalizer` | `implementation`, `norm_method`, `n_jobs` |

The three independent filters share the `FeatureFilter` section; do not create configuration sections named after their individual classes. Their constructor/run-time overrides accept only their relevant field subsets, listed below.

The table lists stored schema fields. For imputation, prefer the input aliases `r_route_method` and `s_route_method` in constructors, run-time overrides, and TOML/JSON configuration. They resolve to `mar_method` and `mnar_method`, respectively; `model_dump()` retains these historical field names for compatibility. Conflicting aliases in the same input layer are rejected. `mnar_fraction` and the filtering thresholds named `mnar_*` remain supported field names; they configure S-route handling and do not imply a missingness mechanism.

`global_seed` is configured under `Dataset`; correction, imputation, and normalization also accept it as a direct constructor/run-time override. It is not an extra field in those three TOML/JSON sections.

### 1.3.4 Method and threshold conventions

| Setting | Canonical choices or interpretation |
| --- | --- |
| QA `corr_method` | `Spearman`, `Pearson` |
| QA `scaling_method` | `Auto-scaling`, `Pareto-scaling`, `None`; applied locally for assessment |
| QA reference outlier thresholds | Float from 0 to 1 means a failing-reference ratio; integer ≥ 1 means an absolute count. `1.0` and `1` therefore have different meanings |
| Dataset `boundary` | `IQR` or `sigma` |
| Correction `base_est` | `Auto`, `QC-RLSC`, `robust QC-RLSC`, `QC-RFSC`, `QC-SVR`, `SERRF`, `RUV-III`, `WaveICA 2.0`, `Metanorm-rLOESS` |
| Correction `rlsc_span_selection` | `fixed` or constrained-grid `gcv`, for standard raw-scale QC-RLSC only |
| Correction `rloess_span_selection` | `gcv` (continuous GCV, default) or `fixed`, for MetaNorm-style robust QC-RLSC; `rloess_span` controls fixed fits and `rloess_iterations` controls robust fitting |
| Imputation `s_route_method` (stored as `mnar_method`) | `QRILC`, `Row-wise`, `Column-wise`, `Global` |
| Imputation `r_route_method` (stored as `mar_method`) | `Auto`, `MinProb`, `KNN`, `LLS`, `BPCA`, `Median` |
| Normalization `norm_method` | `Auto`, `ROBUST_LOG_ONLY`, `TIC`, `Median`, `PQN`, `MDFC`, `Quantile`, `VSN` |

Method dispatch normalizes supported spellings and may record a canonical identifier such as `MEDIAN` or `QUANTILE`. Use `selected_method` for programmatic decisions and `selected_label` for display where provided.

### 1.3.5 Optional R implementations

`implementation="python"` is the default for correction, imputation, and normalization. Selecting `"r"` changes the scientific implementation, not the Joblib execution backend. Configure it using the same constructor, run-time, or file precedence rules as other stage settings. R supports compatible explicit methods and `AUTO`; implicit cross-language fallback is never used.

AUTO keeps the existing candidate sets and scoring definitions. Providers are assigned before evaluation; this is not a competition between both implementations of every method.

| Stage | Candidates replaced by original R | Candidates retaining Python |
|---|---|---|
| Correction | Robust QC-RLSC, SERRF, RUV-III, WaveICA 2.0 | Standard QC-RLSC, QC-SVR |
| Imputation | BPCA, QRILC | KNN, MinProb, Median, LLS |
| Normalization | VSN | ROBUST_LOG_ONLY, PQN, MDFC, TIC, Median, Quantile |

An incompatible or failed R AUTO candidate remains failed, while eligible alternatives continue; the same method is not retried in Python. Explicit unsupported R/method combinations still raise errors. `selection["requested_implementation"]` retains the request, `selection["implementation"]` identifies the selected method's actual backend, and candidate rows retain their own `implementation` and `implementation_provenance`. Imputation additionally records `s_route_implementation`, with `mnar_implementation` retained as a compatibility key. Repeated AUTO runs retain the original request even when a Python-only candidate wins.

```python
from pimqc import DataNormalizer, MissingValueImputer, SignalCorrector

# Choose only the stages for which you want original R execution.
imputed_result = MissingValueImputer(
    quality_filtered_dataset,
    pipeline_params=params,
    r_route_method="BPCA",
    s_route_method="QRILC",
    implementation="r",
).run_imputation()

normalized_result = DataNormalizer(
    imputed_result.data,
    pipeline_params=params,
    norm_method="VSN",
    implementation="r",
).run_normalization()

# Correction runs earlier in the pipeline, before quality filtering.
corrected_result = SignalCorrector(
    mv_filtered_dataset,
    pipeline_params=params,
    base_est="QC-RLSC",
    rlsc_robust=True,
    implementation="r",
).run_signal_correction()
```

Equivalent TOML settings are:

```toml
[MissingValueImputer]
implementation = "r"
r_route_method = "BPCA"
s_route_method = "QRILC"

[DataNormalizer]
implementation = "r"
norm_method = "VSN"

[SignalCorrector]
implementation = "r"
base_est = "QC-RLSC"
rlsc_robust = true
```

These choices leave the `MetaboDataset`/`StageResult`/typed Audit contract unchanged. Access provenance at `result.audit.metrics["selection"]["implementation_provenance"]`; normalization also exposes the same selection via `result.audit.selection`. Entries contain the actual R package/function, versions, available upstream source metadata, parameters, random seed/RNG kind, transformations, and warnings. No R objects are retained; existing serialization and detached plotting remain usable without R. A skipped imputation has no R invocation and therefore no fabricated execution provenance.

R calls run serially on the main Python thread. R seeds must be integers from 0 to 2,147,483,647; identical integer seeds in Python and R do not imply identical stochastic outputs. The backend restores the host R random state after execution. Missing R/rpy2/packages, invalid upstream inputs, or failed R fits raise an error rather than return an unlabeled Python result. The existing `calc_*` and `impute_by_*` helpers remain native Python computations; the selectable implementation is a processor/stage API. Quantile remains Python-only. WaveICA 2.0, RUV-III, and the pinned-source SERRF adapter are also available for explicit R correction; their requirements are detailed in [1.4.7.1](#1471-original-r-correction-adapters).

Use separate output directories when comparing Python and R runs. Imputation retains its existing method-based CSV/SVG names, and report/audit output locations are not automatic run identifiers.

For installation, the verified local versions, diagnostics, and optional tests, see the [optional R backend guide](r_backend.md). `pimqc.processing.r_backend.check_r_environment(packages=())` reports runtime/package availability without installing anything; the numerical adapter helpers are internal implementation details.

#### 1.3.5.1 Input and output scales

An original R function may perform its own centering, scaling, or generalized-log transformation. Do not pre-log data just because the selected implementation is R. The stage accepts its declared input scale and owns any adapter transformation: imputation uses `log2(x+1)` and `exp2(x)-1`; RUV-III uses the same outer pair; Metanorm rLOESS uses `log2(x)` and `exp2(x)`; VSN receives raw data and returns its intrinsic generalized-log2 output without a second log. WaveICA and SERRF receive raw data but still perform their own internal correction operations.

VSN generalized-log values are not ordinary log2 intensities and cannot be inverted with a generic `exp2(x)-1`. RUV-III may return valid log-space output whose inverse violates the stage's positive raw-intensity contract. The correction boundary converts those raw inverse values to missing and records affected cells; it does not silently recenter, clip, or substitute original measurements. Accurate dataset context is required to detect already transformed inputs. See [original R correction adapters](#1471-original-r-correction-adapters) for the method-specific input and control requirements.

#### 1.3.5.2 Default numeric-domain policies and AUTO evaluation

No additional run-time choice is required. Correction delivers positive raw intensities or missing values; imputation delivers positive reconstructed intensities while preserving observed entries; normalization retains legitimate signed log2 or VSN generalized-log coordinates. These stage-specific policies share inspection utilities but do not apply one positivity rule to every matrix.

Correction preserves feature-level route assignments and adds cell provenance in `feature_metadata["correction_missing_samples"]`, a tuple of sample IDs per feature. Existing provenance is merged. Imputation sends still-missing correction-induced cells through R-route reconstruction even on an S-route feature, leaves original missing cells on S-route features under the configured special-handling strategy, and clears consumed provenance. Normalization uses the completed values, not route labels.

`context.extra_attrs["value_scale"]` records `raw_positive`, `log2`, `log2p1`, `vsn_glog` or `centered_scaled` without changing the artifact schema. QA uses a named positive RSD view: raw intensities, `exp2` for log2, `exp2 - 1` for log2p1, or an `exp2` diagnostic view for VSN. The VSN view is **not an inverse to original intensities**; centered/scaled input has no intensity RSD. Per-feature stable rescaling avoids overflow without changing CV; no epsilon clipping is used.

AUTO evaluates production-policy outputs, independently of Python–R kernel comparison. Shared input support and planned metric weights prevent a candidate from gaining weight by losing observations or diagnostics. Failed candidates retain their reasons and domain diagnostics. Nonconverged native VSN is excluded from AUTO; an explicitly requested finite VSN result remains available with degraded status. Missing dependencies, invalid shapes and model-fitting failures remain errors rather than automatic backend substitutions.

## 1.4 Processing API

### 1.4.1 Stage lifecycle

Recommended entry points are the named `run_*` methods below. They return `StageResult` and follow compute → export → render. With `output_dir=None`, they compute scientific output and audits, including stored plotting diagnostics, without exporting files or drawing figures. Supplying an output directory enables CSV/figure output; it does not automatically create serialized dataset/audit artifacts.

Individual stage runners defer directory creation until computation succeeds. A skipped stage can still export a valid table, but should not render an empty dashboard. The complete pipeline creates its root output directory before running stages, so failure does not imply that no directory exists.

The public build, QA, filtering, correction, imputation, normalization, and pipeline run entry points use execution-time logging. The current decorator logs successful calls only when elapsed time exceeds one second; it includes export/render time when enabled. Timing is not a universal `audit.execution_time` field, and direct compute helpers do not promise the same timing log.

### 1.4.2 `QualityAssessor`

Constructor: `QualityAssessor(data, pipeline_params=None, corr_method=None, scaling_method=None, is_outlier_threshold=None, orf_outlier_threshold=None)`.

| Method | Return type |
| --- | --- |
| `compute_assessment()` | `StageResult[AssessmentDiagnostics]` |
| `run_assessment(output_dir=None, legend_mode="external", context_updates=None, **runtime_overrides)` | Same result, with optional diagnostic exports/rendering |

The four constructor settings are also accepted as run-time overrides. `context_updates` supplies `ProcessingContext` field updates for the assessment input; it is not a general configuration dictionary. `legend_mode` supports `external` and `local` for exported QA panels.

```python
from pimqc import QualityAssessor

qa_result = QualityAssessor(
    raw_dataset,
    pipeline_params=params,
).run_assessment()
qa_diagnostics = qa_result.data
qa_audit = qa_result.audit
```

`AssessmentDiagnostics` contains correlation, PCA, RSD, reference-feature evaluation, and outlier results. It is not a processed dataset; continue processing with the input `MetaboDataset`. An empty dataset produces a skipped QA audit. See [1.5](#15-result-and-audit-api) for audit fields and [1.7](#17-plotting-api) for rendering without recalculation.

### 1.4.3 `SampleMissingValueFilter`

Constructor: `SampleMissingValueFilter(data, pipeline_params=None, *, sample_mv_tol=None)`.

| Method | Return type |
| --- | --- |
| `filter_samples_by_missingness()` | `StageResult[MetaboDataset]` |
| `run_filter_samples_by_missingness(output_dir=None, **runtime_overrides)` | Same result, with optional CSV exports |

Only `sample_mv_tol` is an allowed run-time override. This action screens pooled-QC and actual samples by sample-level missingness, preserving Blank and other non-target roles. Its `SampleFilterAuditPayload` contains sample tracking, dropped sample IDs, execution status, and counts. It never produces a standalone single-brick dashboard; its sample diagnostic is incorporated into the feature MV dashboard when the result is explicitly supplied.

### 1.4.4 `FeatureMissingValueFilter`

Constructor: `FeatureMissingValueFilter(data, pipeline_params=None, *, sample_result=None, mv_global_tol=None, mv_group_tol=None, mv_qc_tol=None, mnar_group_mv_tol=None, mnar_qc_mv_tol=None, mnar_intensity_pct=None)`.

| Method | Return type |
| --- | --- |
| `filter_features_by_missingness(*, sample_result=None)` | `StageResult[MetaboDataset]` |
| `run_filter_features_by_missingness(output_dir=None, *, sample_result=None, **runtime_overrides)` | Same result, with optional CSV exports and combined MV dashboard |

The six threshold fields are also allowed run-time overrides. `sample_result` is explicit provenance, not an instruction to run sample filtering again. The input `data` should be the corresponding sample-filtered dataset if that step was executed.

```python
from pimqc import FeatureMissingValueFilter, SampleMissingValueFilter

sample_result = SampleMissingValueFilter(
    raw_dataset,
    pipeline_params=params,
).run_filter_samples_by_missingness(output_dir="outputs/02_MV_Filtered")

feature_mv_result = FeatureMissingValueFilter(
    sample_result.data,
    pipeline_params=params,
    sample_result=sample_result,
).run_filter_features_by_missingness(output_dir="outputs/02_MV_Filtered")
mv_filtered_dataset = feature_mv_result.data
```

Features are assessed using global, QC, and available biological-group missingness, then assigned to R-route or S-route or excluded. R-route retains features meeting the baseline missingness limits. S-route retains features through a group-pattern rule or low-intensity QC rescue. Canonical labels travel in `result.data.feature_metadata["imputation_route"]` as `R-route` or `S-route`; the parallel historical `missingness_type` column retains `MAR`/`MNAR` solely for compatibility. The full retention history is in `result.audit.feature_tracking`, whose `Stage1_Status` records current route labels and any group/QC rescue detail.

Running without `sample_result` is supported: metrics record `sample_filter_status="not_run"` and the dashboard omits the sample-filter panel. Without biological groups, group-specific routes/panels are unavailable. With no missing values, feature MV is marked skipped. These conditions can coexist; inspect recorded execution status and available diagnostics rather than assuming every panel exists.

The shared directory contains `Filtered_Data_High-MV_Samples.csv`, `Filtering_Tracking_High-MV_Samples.csv`, `Filtered_Data_MV_Features.csv`, `Filtering_Tracking_MV_Features.csv`, and, when meaningful, `MV_Classification_Dashboard.svg`. Independent execution produces only the artifacts owned by that action.

### 1.4.5 `FeatureQualityFilter`

Constructor: `FeatureQualityFilter(data, pipeline_params=None, *, missingness_metadata=None, qc_rsd_tol=None, blank_qc_ratio_tol=None)`.

| Method | Return type |
| --- | --- |
| `filter_features_by_quality(*, missingness_metadata=None, idx_mar=None, idx_mnar=None, missingness_tracking=None)` | `StageResult[MetaboDataset]` |
| `run_filter_features_by_quality(output_dir=None, *, missingness_metadata=None, idx_mar=None, idx_mnar=None, **runtime_overrides)` | Same result, with optional CSV exports and quality dashboard |

Allowed run-time overrides are `qc_rsd_tol` and `blank_qc_ratio_tol`. Supply `missingness_metadata` as a Series or DataFrame containing recognized R-route/S-route labels; supported columns include `imputation_route`, `Stage1_Status`, and the legacy `missingness_type` column. Historical MAR/MNAR values remain accepted as route aliases. Prefer the complete feature MV audit table, which retains previously excluded features as well as retained labels.

```python
from pimqc import FeatureQualityFilter

# corrected_dataset is the selected output of signal correction.
quality_result = FeatureQualityFilter(
    corrected_dataset,
    pipeline_params=params,
    missingness_metadata=feature_mv_result.audit.feature_tracking,
).run_filter_features_by_quality(output_dir="outputs/04_Quality_Filtered")
quality_filtered_dataset = quality_result.data
```

Both routes undergo the Blank/QC check when blanks are available; the QC-RSD screen applies only to R-route features, with S-route features exempt. Without recognized route labels, the stage runs in `quality_only` mode using the Blank/QC and QC-RSD checks for all eligible features; it does not silently rerun feature routing. Explicit `idx_mar`/`idx_mnar` are retained lower-level API names for R-route/S-route indices but do not reconstruct the complete retention history. Audit tables additionally expose the descriptive aliases `idx_r_route` and `idx_s_route`.

`missingness_tracking` is a compute-method argument, not a supported keyword for `run_filter_features_by_quality()`. For a run lifecycle, pass full tracking through `missingness_metadata` instead.

### 1.4.6 `FilteringOrchestrator` and `FilteringRunResult`

`FilteringOrchestrator(data, pipeline_params=None)` manages hand-offs between the three filters without merging their audits.

| Method | Arguments after `self` | Behavior |
| --- | --- | --- |
| `run_missingness` | `*, output_dir=None, sample_overrides=None, feature_overrides=None` | Runs sample then feature missingness into one directory |
| `run_quality` | `data=None, *, output_dir=None, quality_overrides=None` | Uses saved full MV tracking when available; accepts corrected input data |
| `run` | `*, quality_data=None, missingness_output_dir=None, quality_output_dir=None, sample_overrides=None, feature_overrides=None, quality_overrides=None` | Runs the three filters sequentially; does not execute correction |

Each returns a `FilteringRunResult` with optional `sample_result`, `feature_result`, and `quality_result` fields. `missingness_metadata` returns a copy of retained feature metadata; `missingness_tracking` returns a copy of the full feature MV audit table. Both return `None` if no feature result exists. There is no generic `.data` property on `FilteringRunResult`; select the named result you need.

For the full scientific order, call `run_missingness()`, run correction on `.feature_result.data`, and then call `run_quality(data=corrected_dataset)`. Calling `run_quality()` first is supported and selects `quality_only` behavior.

### 1.4.7 `SignalCorrector`

`SignalCorrector(data, pipeline_params=None, ...)` accepts every `SignalCorrector` field listed in [1.3.3](#133-accepted-fields) as an explicit optional constructor argument, plus `global_seed`. These are named parameters, not an arbitrary options dictionary; all default to `None` at the override boundary. `implementation` defaults to `"python"`.

`run_signal_correction(output_dir=None, **runtime_overrides) -> StageResult[dict[str, MetaboDataset]]` returns an ordered mapping of the selected method's correction stages. It does not return every Auto candidate's intensity matrix. Algorithm/seed settings can also be overridden at run time.

```python
from pimqc import SignalCorrector

correction_result = SignalCorrector(
    mv_filtered_dataset,
    pipeline_params=params,
    base_est="QC-RLSC",
).run_signal_correction()
corrected_stages = correction_result.data
corrected_dataset = list(corrected_stages.values())[-1]
```

QC-based correction can produce `Intra-batch corrected` and, when applicable, `Inter-batch corrected` datasets. Global-method outputs use labels such as `SERRF`, `RUV-III`, or `WaveICA 2.0`. Select the last output for downstream quality filtering; assess each output separately when comparing correction stages.

With `base_est="robust QC-RLSC"` or `base_est="QC-RLSC", rlsc_robust=True`, the revised robust method follows MetaNorm rLOESS: positive raw intensities are logged with `log2`, quadratic symmetric LOESS is fitted within each batch, and residuals are restored to the reference log-mean before `exp2`. It is no longer the earlier fixed-span, raw-intensity robust ratio correction. Standard QC-RLSC (`rlsc_robust=False`) retains its existing algorithm and parameters. The robust method instead uses `rloess_span_selection="gcv"`, `rloess_span=0.75` for explicit fixed-span fits, and `rloess_iterations=4` (total fits, including the initial fit, as in R). These replace the former public `rlsc_robust_iterations` setting; existing configurations must be migrated rather than silently interpreted as the new algorithm. Its Python backend uses the compiled scikit-misc LOESS kernel without requiring R; `implementation="r"` selects the original R LOESS fitting kernel with the same declared fitting rules.

The robust Python/R paths use the same finite-sample support checks and refit span selection and the model using training QCs inside each validation fold. Held-out QC responses do not determine the restoring mean. Neither backend extrapolates outside the training-QC injection range; unsupported cells and failed fits remain missing and contribute to coverage diagnostics. The production adapter excludes Blank samples from fitting/reference statistics and keeps their measurements unchanged. These missingness, support, and held-out-validation rules must be distinguished from an untouched full-input `metanormWorker` call. No full-fit predictions are substituted for unavailable OOF predictions.

Robust reweighting follows the original six-MAD bisquare rule, with weighted local-design checks before each fit. This explicit loop prevents singular sparse-QC fits from entering the unsafe native symmetric-fit path; the R adapter applies the same support rules. The fitting kernels and adapter-level safeguards are recorded separately from the original MetaNorm reference. Insufficient support is an unavailable fit, not a request to change polynomial degree, extrapolate, substitute raw measurements, or switch language backends.

Native robust LOESS runs in one isolated worker process; `n_jobs` does not make these fits multithreaded. This contains compiled-kernel failures without terminating the notebook kernel. A native runtime failure disposes of the worker and fails the candidate, never retries the same method in R, and never substitutes raw data. The R backend remains on the main Python thread. Shared weighted-design checks require at least five span neighbours; continuous GCV is restricted to spans at most `0.95`, so five QCs alone cannot support its default search. Even larger QC counts do not guarantee estimability after robust reweighting or fold splitting. Consult the fit diagnostics and held-out coverage, not only the nominal sample count.

Native SERRF fits per batch, intersects QC and biological-sample correlation rankings, standardizes predictors within each role, and performs role-specific cross-batch alignment. QC validation refits predictor selection, scaling and forests inside each fold; biological-cohort statistics remain transductive and are not a prediction model for a new cohort. The low-level `SERRFCorrector.fit_transform` no longer accepts a precomputed pooled `corr_mat`: that matrix cannot represent batch/fold-specific fitting. Native diagnostics distinguish full and fold fits, unavailable predictors, unsupported fits, invalid denominators and OOF coverage. Nonpositive or nonfinite prediction denominators produce missing full-output values; invalid held-out QC predictions remain unavailable rather than substituting training or original values. The same final correction-domain policy applies to both backends, while any repairs internal to the original R function remain separately documented.

Auto compares its registered candidate set, including standard/robust QC-RLSC, QC-SVR, SERRF, RUV-III, and WaveICA 2.0. Its score has three fixed-support components: technical precision (the equal-weight combination of median and feature-wise QC-RSD improvement), biological-variation preservation (canonical D-ratio improvement), and sample-structure preservation. When all three are supported by the input, their weights are 0.35, 0.35, and 0.30. An input-unsupported component is omitted and the remaining weights are renormalized identically for all candidates. A candidate-specific unavailable metric instead contributes zero without shrinking the denominator. The two technical-precision terms follow the same fixed-input-support rule.

Candidates use held-out QC (OOF) evidence where available; jointly fitted WaveICA 2.0 and RUV-III use descriptive full-model evidence. Robust QC-RLSC and R SERRF require usable OOF evidence for selection and cannot substitute full-fit metrics. Other candidates retain a descriptive full-model fallback when OOF evidence is absent. Read each candidate's `validation` passport, especially `evaluation_basis`, rather than assuming that every score is held out. Not every explicitly supported method is an Auto candidate. Requested/selected methods, candidate evidence, and selected predictions remain available for plotting without rerunning fitting or cross-validation.

The candidate comparison dashboard places QC-RSD and D-ratio distributions side by side. Each candidate has one box per metric, using its final correction stage and the recorded evaluation basis; intermediate stages and duplicate full-fit/OOF boxes are omitted. Short method labels retain `Inter-batch` where applicable, and the shared legend below both panels distinguishes OOF from full-model evidence. Ineligible candidates or missing evaluation distributions remain N/A rather than being replaced with descriptive full-fit results. OOF denotes held-out QC evaluation, not independent validation of every study sample. These distributions accompany, but are not identical to, the composite selection scores. Detailed selected-method dashboards retain their existing stage-by-stage diagnostics.

Set `implementation="r"` together with `base_est="Metanorm-rLOESS"` to call the exported `metanorm::metanormWorker` through `rpy2`. This is a distinct method, not an interchangeable R alias for QC-RLSC; it is not included in `AUTO`. The wrapper calls the original worker sequentially because the upstream high-level `metanorm()` starts a PSOCK cluster even with one CPU. It uses positive unlogged intensities, `log2 → rLOESS → exp2`, QC-only fitting, per-batch GCV, and `keepScale=TRUE`, without adding pseudocounts or extra batch alignment. The original full-input log-mean centering includes Blank columns when present. At least four distinct finite QC injection orders per feature/batch are required, and every observed target must fall within that feature's QC order range; the original LOESS cannot extrapolate. Invalid inputs or fits raise an error. OOF is explicitly unavailable, full-fit QC diagnostics are descriptive, and no AUTO score is fabricated. Other methods' parameters such as `loess_span`, `rlsc_span_grid`, and `cv_folds` do not tune this R worker.

AUTO records candidate execution errors in `audit.metrics["selection"]["failed_candidates"]` and continues evaluating alternatives. Computed candidates lacking required selection evidence remain in `audit.metrics["selection"]["candidate_results"]` with `status="ineligible"` and validation details. Selection raises an error if every candidate fails or no candidate is eligible; an explicit fixed-method execution error is propagated. The selected `validation` passport reports OOF availability, requested/effective folds, QC-value coverage, and the evaluation basis. A descriptive full-model fallback is not labeled as cross-validation. Global-model QC RSD is not independent validation. Reusing an AUTO processor retains the AUTO request.

#### 1.4.7.1 Original R correction adapters

R SERRF executes the verified author function with an audited NA-safe subscript fix (`runtime_fixes`) and per-forest seed adaptation (`runtime_seed_adaptation`); the local source and checksum are unchanged. The indexing fix prevents zero/negative repair selectors containing NA from causing an assignment error, while retaining the original replacement distributions. Features with no observed value in any fitting batch cannot establish the required batch minimum: `unsupported_feature_ids` and `all_missing_features_by_fit_batch` record their exclusion, and their non-Blank output remains unavailable across batches. The same rule is recomputed using training QCs and real biological samples inside each OOF fold, never held-out QC values. Supported complete inputs retain the original numerical path apart from the explicit seed adaptation. These explicit runtime adaptations qualify references below to the original function and its stochastic repair policy.

Select one of the following explicit `base_est` values, or `base_est="Auto"`, together with `implementation="r"`. AUTO replaces the corresponding candidate kernels with R while retaining the original candidate set; selecting R is not a claim of cross-language numerical equivalence. All three adapters exclude Blank columns from fitting, preserve their values unchanged, and require unlogged, unscaled intensities. WaveICA/RUV-III share a temporary feature-median input adapter with their Python counterparts, since the original kernels require complete matrices. Medians use only observed non-Blank values, in raw space for WaveICA and log space for RUV; the original missing mask is restored after correction. `missing_input_adapter` diagnostics record the scale, counts and reference population. No additional option is needed. Fully missing fitting features and insufficient real RUV replicate-control support are rejected rather than filled with invented references. SERRF accepts NaN and zeros for its unchanged upstream stochastic repair, records input counts and repair ownership, and rejects negative/infinite input. Its adapter restores original non-Blank NA positions in full-fit and OOF outputs before the production correction-domain policy, preserving the later imputation stage's responsibility. Provenance and diagnostics record `output_missing_policy="restore_original_missing_positions"` and `restored_missing_count`, the number of original non-Blank NA cells. Zero inputs are not additionally converted to NA by this restoration. Temporary repairs still influence model fitting and other corrected values; restoring NA does not remove that influence or guarantee support for arbitrary missingness. Upstream failures remain failures; no pipeline reorder occurs.

| Method | Original entry point | Effective settings and input requirements |
| --- | --- | --- |
| `WaveICA 2.0` | `WaveICA2.0::WaveICA_2.0` | `waveica_components` maps to `K`, `waveica_cutoff` to `Cutoff`, and R-only `waveica_alpha` to `alpha` (default `0`). Uses `wf="haar"`, the upstream wavelet depth and GAM defaults, and preserves the supplied finite intensity scale. Requires at least 10 non-Blank samples, distinct globally ordered injection values, and `K >= 2`; the original package limits `K` to the smaller matrix dimension. Sorting is restored on output. Native `waveica_levels`, `waveica_spline_knots`, and `waveica_max_iter` do not tune R and non-default values are rejected. |
| `RUV-III` | `ruv::RUVIII` | Explicit nonempty `ruv_control_features` supplies feature IDs; `ruv_k` supplies `k`. No empirical low-RSD controls are automatically substituted. The adapter applies `log2(x+1)`, calls R on samples × features, then reconstructs with `exp2(x)-1`; invalid reconstructed intensities are recorded and converted to missing at the correction boundary rather than clipped. This is standard RUV-III, not the missing-value variant RUV-III-C. |
| `SERRF` | Pinned Shiny-SERRF `serrfR`, using `ranger` | `serrf_r_source` points to the separately obtained original `app.R` or its verified extraction. `serrf_corr_features` maps to `num` (at least 2, less than the feature count); the native `serrf_corr_features=0` order-only mode has no original-R equivalent. Passes NaN/zeros to original repairs; negative/infinite input is rejected. Requires at least 3 QCs and 3 biological samples in every batch; constant, fully positive role/batch vectors are rejected. Missing-data fitting can still fail upstream. Native `serrf_n_tree`, `serrf_backend`, and `serrf_batch_size` do not tune this function and non-default values are rejected. |

For RUV-III, the default repeat design assumes all supplied QCs are aliquots of the same pooled material and treats each other non-Blank sample as independent. To use true technical replicates or distinct QC pools, set `ruv_replicate_column` to a sample-metadata column whose shared labels identify the same underlying material. This must not be a biological treatment/group column. Every non-Blank sample needs a label, including unique labels for unreplicated samples. The adapter validates the repeat design, negative controls, and available rank rather than silently reducing `k`. Internal standards can be explicit controls only when their scientific interpretation supports the negative-control assumption.

Example for an already complete raw-intensity dataset (replace the example IDs with scientifically justified controls):

```python
ruv_result = SignalCorrector(
    complete_dataset,
    base_est="RUV-III",
    implementation="r",
    ruv_control_features=["IS_1", "IS_2", "IS_3"],
    ruv_k=2,
).run_signal_correction()

waveica_result = SignalCorrector(
    complete_dataset,
    base_est="WaveICA 2.0",
    implementation="r",
    waveica_components=5,
    waveica_cutoff=0.1,
    waveica_alpha=0.0,
).run_signal_correction()

serrf_result = SignalCorrector(
    complete_dataset,
    base_est="SERRF",
    implementation="r",
    serrf_r_source="/path/to/pinned/Shiny-SERRF/app.R",
    serrf_corr_features=10,
).run_signal_correction()
```

`pimqc.processing.correction.serrf_r.prepare_serrf_source(source_path, output_path) -> pathlib.Path` verifies the pinned author file, extracts only its unchanged `serrfR` function using R syntax parsing, and saves a reusable local function file. The adapter never executes the Shiny application or arbitrary top-level source expressions. At execution, UI callbacks are isolated and the verified function receives recorded NA-indexing and per-forest seed adaptations in memory. The source file/checksum and upstream `ranger` defaults remain unchanged; native Python forest options do not replace them. See the [README](r_backend.md#4-prepare-the-optional-local-serrf-source) for the supported source commit, checksum, dependency installation, and redistribution caveat.

The upstream function calls `set.seed(1)` internally for every forest. The runtime AST adapter verifies that this is the sole seed call and replaces only its constant with `random_state` in both full fit and OOF refits. For example, `random_state=123` gives an effective per-forest seed of 123. Provenance records the outer seed, upstream `original_model_seed=1`, `effective_model_seed`, and `runtime_seed_adaptation`; it identifies source-adapted execution, not an unmodified original function. The surrounding R RNG state is restored after the call. Tree count, predictors, and other algorithm settings are unchanged; later stochastic repairs may follow a different RNG trajectory while retaining the upstream replacement distributions.

R SERRF uses the same NA-safe source adapter for the full fit and an explicit pi-metaboqc held-out-QC validation layer. With `cv_folds` enabled, each fold refits the function using training QCs and the real biological samples only. The wrapper captures the actual ranger models, selected predictors and training-QC scales, predicts excluded QCs, and applies alignment learned solely from training output. Held-out QCs are never reclassified as biological samples. Biological-cohort statistics remain transductive, so this validates within-cohort QC repeatability, not prediction on a new cohort. Failed folds or invalid denominators remain missing and are never replaced by full-fit values. The internal `SERRFRCorrector` adapter defaults to five folds and permits `cv_folds=0` for a full-fit-only reference call, which cannot supply OOF evidence for AUTO. The public `SignalCorrector` constructor, run method and TOML/JSON configuration require at least two folds; their built-in default is five, while the bundled demo explicitly requests three.

Python and R SERRF use the same deterministic within-batch fold assignment. Diagnostics record fold IDs, effective folds, coverage, failures, validation origin, and R full-fit/OOF elapsed times. This is not an OOF API provided by the original author. RUV-III and WaveICA retain full-fit QC-RSD evaluation in AUTO, as their existing native counterparts do; this remains descriptive, not independent held-out validation. Metanorm-rLOESS remains outside AUTO.

The audit records actual R functions, runtime/package versions, transformations, Blank policy, effective options, repeat/negative-control design where applicable, and source identity for the SERRF extraction. Source files and R packages are not needed to load the saved results or render their payloads. Use separate output directories when comparing Python and R runs: correction retains method-based CSV/SVG names.

Original WaveICA may return negative intensities. Low-level adapters retain signed/nonfinite outputs as diagnostic evidence; the production correction stage converts invalid raw intensities to missing before metrics, selection, plotting payloads and delivery. Features or cells lost by a candidate do not disappear from its planned AUTO support. Full and OOF coverage and affected coordinates are recorded in the audit.

For native `RUV-III`, `ruv_control_features` can also explicitly select the controls without automatic augmentation. If omitted, the existing native heuristic combines declared standards/reference features with low-RSD biological features; these are operational controls, not proven biological negative controls. The native QC/control-residual SVD and recentering remain a distinct estimator from the original R call; the implicit intensity floor has been removed. `audit.metrics["selection"]["native_diagnostics"]` records effective controls/rank, `output_negative_count`, `output_zero_count`, `output_nonfinite_count`, and `output_policy`. A custom `ruv_replicate_column` remains specific to the original-R backend.

### 1.4.8 `MissingValueImputer`

Constructor: `MissingValueImputer(data, pipeline_params=None, mar_method=None, mnar_method=None, mnar_fraction=None, knn_neighbors=None, lls_neighbors=None, bpca_components=None, bpca_max_iter=None, bpca_tol=None, sim_mask_ratio=None, global_seed=None, implementation=None, *, r_route_method=None, s_route_method=None)`.

`run_imputation(output_dir=None, **runtime_overrides) -> StageResult[MetaboDataset]` accepts the same algorithm/seed settings as run-time overrides.

```python
from pimqc import MissingValueImputer

imputation_result = MissingValueImputer(
    quality_filtered_dataset,
    pipeline_params=params,
    r_route_method="KNN",
).run_imputation()
imputed_dataset = imputation_result.data
```

The imputer reads `feature_metadata["imputation_route"]`, falling back to historical `missingness_type` when needed. Without either column it uses R-route. S-route uses QRILC or the configured row/column/global small-value fraction strategy; the fraction is not a measured detection limit. R-route uses the requested reconstruction method or automatic selection. Only QC/actual target columns are filled; Blank columns are preserved.

Route labels are stripped and matched case-insensitively, then normalized to `R-route` or `S-route`; historical `MAR` and `MNAR` are accepted only as aliases. Unknown/null labels raise an error, as do conflicting canonical and legacy assignments. These are operational feature categories, not inferred missingness mechanisms. S-route-only execution records reconstruction selection as `Not required` unless correction-induced gaps still require R-route processing. R-route imputation first uses QC and actual-sample evidence separately; unresolved values may use pooled evidence. `audit.metrics["isolation_fallbacks"]` records cross-role fills in the final transformation, excluding benchmark simulations. A matrix still containing missing target values is not returned as completed.

Automatic reconstruction metrics come from intensity-aware masked-value challenges, including low-intensity-weighted masking based on the fitted intensity distribution. This tests reconstruction under a demanding abundance pattern; it does not simulate or identify MAR/MNAR statistical mechanisms. The metrics are not identical to final-data distribution/structure metrics. `audit.candidate_results` stores method-specific metrics and true/predicted benchmark values. Keep these with the audit rather than recomputing them in a plotter.

QRILC participates in the reconstruction benchmark, but `r_route_method="QRILC"` is not an accepted explicit setting in the current public schema. Use `r_route_method="Auto"` for candidate selection and `s_route_method="QRILC"` for the separate special-handling route; do not infer valid configuration values from candidate labels alone.

Native QRILC uses the actual one-dimensional sampling SD of the inspected original implementation (`fitted_slope * tune_sigma`), not its square root. Its low-level kernel retains signed log-scale draws and warns when a small or constant column uses the native constant-fill extension. Both production backends use the same post-kernel repair: only newly imputed finite nonpositive predictions are replaced by half the smallest positive raw training observation from the same feature and sample role, falling back to the same role across features. Nonfinite predictions and inverse overflow still fail. Masked truth, previously imputed cells and other roles never determine the replacement; observed values and legitimate small positive predictions are unchanged. Stochastic imputation should be evaluated on held-out entries and within-column distributions relative to repeated R runs, not by full-matrix correlation dominated by unchanged observations.

Set `implementation="r"` with `r_route_method="BPCA"` or `"Auto"`, and `s_route_method="QRILC"`. The stage calls `pcaMethods::pca`/`completeObs` and `imputeLCMD::impute.QRILC` for the corresponding evaluation and final-filling operations. Automatic selection reuses the same masked-value benchmark and six reconstruction candidates, replacing BPCA/QRILC with R while retaining Python for the other methods. BPCA receives `bpca_components`, `bpca_max_iter`, and `bpca_tol`, with upstream centering enabled and scaling disabled; QRILC uses `tune.sigma=1`. The workflow supplies `log2(x + 1)` and reconstructs with `exp2(x) - 1`; already logged/scaled input is rejected. Finite nonpositive new predictions use the common training-only half-minimum repair, whereas nonfinite predictions or inverse overflow remain failures. R-route models are fitted separately for QC and biological samples; S-route QRILC receives all target sample roles. Completely missing features or samples within an R input slice are rejected rather than replaced with Python heuristics. Unsupported explicit route-method settings are rejected. Each actual R invocation is recorded in `audit.metrics["selection"]["implementation_provenance"]`, with phase, sample roles, dimensions, and transformations; candidate rows separately retain their evaluation/final-fit provenance. The `R` in R-route describes reconstruction and is independent of `implementation="r"`.

If no target values require imputation, `audit.skipped` is true, `selected_method` is `"Not required"`, `is_auto` is false, and `plot_payload` is `None`. With exports enabled the unchanged result is written as `Imputed_Data_NotRequired.csv`, without a dashboard. A run with no R-route features and no correction-induced reconstruction gaps also omits reconstruction dashboards even if S-route imputation was performed.

### 1.4.9 `DataNormalizer`

Constructor: `DataNormalizer(data, pipeline_params=None, norm_method=None, global_seed=None, n_jobs=None, implementation=None)`.

`run_normalization(output_dir=None, **runtime_overrides) -> StageResult[MetaboDataset]` accepts `norm_method`, `global_seed`, `n_jobs`, and `implementation` overrides.

```python
from pimqc import DataNormalizer

normalization_result = DataNormalizer(
    imputed_dataset,
    pipeline_params=params,
    norm_method="VSN",
).run_normalization()
normalized_dataset = normalization_result.data
selection = normalization_result.audit.selection
assert selection["requested_method"] == "VSN"
assert selection["is_auto"] is False
```

TIC, Median, PQN, and MDFC apply robust Log2 after scale normalization; Quantile operates on a robust-logged view; `ROBUST_LOG_ONLY` applies robust Log2 alone. VSN uses its intrinsic generalized-log transform, not an additional external robust Log2 step. Blank samples are excluded from normalized output, and dataset context records transformation state.

`audit.selection` contains requested/selected methods, selected label, Auto flag, and available selection evidence. Candidate results are present for Auto comparison, not fabricated for a fixed method. `audit.output_suffix` controls filenames such as `Normalized_Data_PQN_Log2.csv` and `Normalization_Dashboard_PQN_Log2.svg`; Auto candidate rows may also be exported to `Normalization_Auto_Summary.csv`.

Native VSN fits sample-specific offsets and positive slopes using a profile likelihood and intensity-stratified least-trimmed-squares refinement. `audit.metrics["vsn_parameters"]` retains sample IDs/coefficients, actual glog output offset, fit feature count, convergence and per-round optimizer diagnostics. `vsn_scale` summarizes the geometric mean of sample slopes; `vsn_shift` is the additive glog output offset, not the previous empirical alignment shift. Do not interpret them as a shared-scale calibration. Nonconverged finite fits issue a warning and retain their actual status.

Set `implementation="r"` with `norm_method="VSN"` or `"Auto"` to call `vsn::vsn2` and `vsn::predict`. AUTO retains all existing strategies and scoring, replacing only VSN with R. The R path requires unlogged, unscaled input; neither backend adds an empirical abundance-alignment shift or a second log transform. Blank columns are excluded before the R fit. R model parameters are not misreported as native Python `vsn_scale`/`vsn_shift`; selected execution provenance is retained in `audit.selection["implementation_provenance"]`, with other executed R candidates recorded in `candidate_results`. If R VSN wins, output filenames use `VSN_R`, for example `Normalization_Dashboard_VSN_R.svg`; a Python-only winner retains its native suffix.

Upstream VSN fitting constraints are retained: the verified vsn 3.78.1 matrix method defaults to `minDataPointsPerStratum=42L` and may reject small or unsuitable matrices. The R adapter does not silently lower that bound or change model defaults to force a fit. Native VSN retains its documented smaller-input extension (at least three observed features per sample with nonconstant values); small-input acceptance is not a claim of matching the original package's input policy.

Request configuration is separate from completed-output metadata. Reading `normalization_metrics` before execution does not freeze pending values, and repeated AUTO runs remain AUTO. The payload includes computed RLE summaries and paired Wilcoxon p-values as well as variance/structure diagnostics; rendering does not repeat these calculations.

## 1.5 Result and audit API

### 1.5.1 `StageResult`

`StageResult(data, audit)` is a generic dataclass. Its constructor requires an `AuditPayload` instance; `require_audit(audit_type)` returns the audit narrowed to the requested class or raises `TypeError`.

```python
from pimqc.processing import MissingValueFilterAuditPayload

mv_audit = feature_mv_result.require_audit(MissingValueFilterAuditPayload)
tracking = mv_audit.feature_tracking
metrics = mv_audit.metrics
```

The generic parameter describes `data`, not the audit type. Building, filtering, imputation, and normalization return `MetaboDataset`; QA returns `AssessmentDiagnostics`; correction returns a dataset mapping. Consumers must not assume every `.data` can be passed directly to the next processor.

### 1.5.2 Shared and stage-specific audit fields

All audits provide `metrics`, `audit_type`, `schema_version`, and `contract_identity()`. The latter returns `{"audit_type": ..., "schema_version": ...}`. Audit schema version `"1.0"` is independent of package version `1.5.0`.

There is no universal `status`, `skipped`, `parameters`, or `execution_time` attribute on `AuditPayload`. Status, thresholds, parameters, and decisions are represented by the concrete audit's fields and metric structure.

| Audit type | Principal stage-specific fields/accessors |
| --- | --- |
| `DatasetAuditPayload` | `metric_values`, optional `plot_payload` |
| `AssessQualityAuditPayload` | `outliers`, `qc_correlation`, `batch_qc_correlation`, `pca_results`, `rsd_distribution`, internal-standard/outlier-reference flags and data, metadata-column/role labels, `qc_batches`, `qc_correlation_mask`, `skipped`, optional `plot_payload` |
| `SampleFilterAuditPayload` | `sample_tracking`, `dropped_sample_ids`, `execution_status`, `skip_reason`, `missing_value_count`, `input_sample_count`, `output_sample_count`, `plot_payload` |
| `MissingValueFilterAuditPayload` | `feature_tracking`, `sample_filtered_data`, `sample_tracking`, `qc_mask`, `valid_biological_groups`, `tables`, `execution_status`, `skip_reason`, `missing_value_count`, `plot_payload` |
| `CorrectionAuditPayload` | `requested_method`, `selected_method`, `selected_label`, `is_auto`, metadata labels, `plot_payload`; `candidate_results` and `selected_prediction` forward to the payload |
| `QualityFilterAuditPayload` | `feature_tracking`, `tables`, `plot_payload`; filter mode and execution details are recorded in `metrics` |
| `ImputationAuditPayload` | `candidate_results`, `requested_method`, `selected_method`, `selected_label`, `is_auto`, `mar_feature_count`, `has_candidate_cache`, `skipped`, optional `plot_payload` |
| `NormalizationAuditPayload` | `output_suffix`, `plot_payload`; `selection` and `candidate_results` access stored selection information |

Most classes expose `metric_values` through `metrics`; the sample audit constructs report-facing metrics from tracking/status fields. Preserve typed audits rather than flattening them into generic dictionaries: report metrics alone cannot recreate every figure.

`ImputationAuditPayload.r_route_feature_count` is the descriptive accessor for the serialized compatibility field `mar_feature_count`. Imputation metrics expose `r_route_count`/`s_route_count`, `s_route_method`/`s_route_fraction`, and route definitions while retaining the older keys so saved audits remain readable. Historical plotting payload names such as `mnar_group_mv_tolerance` likewise retain their storage contract; current plots and reports display R-route/S-route.

### 1.5.3 Missing, skipped, and degraded results

| Situation | Consumer behavior |
| --- | --- |
| Sample filter not executed | Retain `not_run` provenance; do not infer sample decisions from the feature-filtered matrix |
| Feature MV has no missing values | Respect skipped status; do not reserve an empty classification dashboard |
| Biological groups unavailable | Omit group-dependent routing/diagnostics; retain available QC/global evidence |
| Feature quality has no recognized MV tracking | Respect `quality_only`; do not infer R-route/S-route assignments |
| Imputation has no missing target values | Preserve skipped audit and unchanged data; no empty dashboard |
| QA skipped | Do not pass a missing payload to `AssessmentPlotter`; `render_assessment()` handles skipped audits directly |
| Auto candidate evidence absent | Do not invent candidate scores or assume a fixed-method run was Auto |

## 1.6 Pipeline API

### 1.6.1 `run_pipeline`

`run_pipeline(meta_df, int_df, params, output_dir=None) -> PipelineResult` takes raw metadata, a feature-by-sample intensity table, and configuration. Its argument names differ from builder/processor constructor names.

```python
from pimqc import run_pipeline

pipeline_result = run_pipeline(
    meta_df=metadata,
    int_df=intensity,
    params=params,
    output_dir=None,
)
final_dataset = pipeline_result.data
```

Without `output_dir`, all scientific stages and QA checkpoints still run, but CSV/figure/report output is disabled. `report_input` remains available. With an output directory, the pipeline exports stages and compiles reports; PDF export can require external dependencies (see [1.9.3](#193-narrative-and-document-export)).

### 1.6.2 `PipelineResult` members and keys

| Member | Contents |
| --- | --- |
| `data` | Property returning `stage_tables["normalized"]` |
| `stage_tables` | Datasets keyed by `raw`, `sample_mv_filtered`, `high_mv_filtered`, `corrected/<stage label>`, `quality_filtered`, `imputed`, `normalized` |
| `stage_results` | Results keyed by `raw_dataset`, `sample_missing_value_filtering`, `high_mv_feature_filtering`, `signal_correction`, `low_quality_feature_filtering`, `missing_value_imputation`, `normalization` |
| `filtering_results` | Explicit dataclass field containing results keyed by `sample_missing_value_filtering`, `feature_missing_value_filtering`, `feature_quality_filtering`; retained by `dataclasses.replace()` |
| `assessments` | QA results keyed by `raw_dataset`, `high_mv_feature_filtering`, `signal_correction/<stage label>`, `low_quality_feature_filtering`, `missing_value_imputation`, `normalization` |
| `pipeline_metrics`, `qa_metrics` | Report-facing mappings assembled from actual audits |
| `report_input` | Structured `ReportInput`, including generated asset manifest when available |
| `output_dir` | Root `Path`, or `None` |
| `report_generated` | Boolean result of document export, not a scientific pass/fail flag |

The sample result is available through both `filtering_results` and `stage_results["sample_missing_value_filtering"]`. The two feature-filter result keys in `stage_results` match the report metric keys. Separate in-memory audits do not imply separate Step 02 directories.

### 1.6.3 Output directories

| Directory | Main contents |
| --- | --- |
| `01_Raw_Data` | Built dataset CSV and acquisition overview |
| `02_MV_Filtered` | Sample/feature MV tables and combined `MV_Classification_Dashboard.svg` |
| `03_Corrected_Data` | Selected correction-stage datasets and diagnostics |
| `04_Quality_Filtered` | Quality-filtered matrix, tracking table, and dashboard |
| `05_Imputation` | Imputed matrix and meaningful reconstruction/candidate dashboards |
| `06_Normalized_Data` | Normalized matrix, dashboard, and optional Auto summary |
| `QA_01_Raw_Data`, `QA_02_MV_Filtered`, `QA_03_Corrected_Data/<stage label>`, `QA_04_Quality_Filtered`, `QA_05_Imputed_Data`, `QA_06_Norm_Data` | Observational QA exports |
| `07_Report_Summary` | Comprehensive/brief narratives, `Report_Input.json`, comparison assets, and requested document exports |

These are pipeline defaults, not mandatory names for individual processors. Custom workflows should keep report paths consistent with generated artifacts.

## 1.7 Plotting API

### 1.7.1 Payload-only construction

Plotters require their typed payload, not a processor instance. Payloads contain detached dataset snapshots and previously computed diagnostics. `snapshot_dataset()` and `snapshot_plot_value()` support detached plotting inputs; they do not run scientific analysis.

| Plotter | Required payload | Principal dashboard entry points |
| --- | --- | --- |
| `DatasetPlotter` | `DatasetPlotPayload` | `plot_dataset_dashboard()` |
| `AssessmentPlotter` | `AssessmentPlotPayload` | `plot_assessment_dashboard(...)`; prefer the audit renderer below for the full argument mapping |
| `FilteringPlotter` | `FilteringPlotPayload` | `plot_mv_filtering_dashboard(...)`, `plot_quality_filtering_dashboard()` |
| `CorrectionPlotter` | `CorrectionPlotPayload` | `plot_correction_dashboard(results_store, selected_method, include_auto_summary=True)`, candidate/article variants |
| `ImputationPlotter` | `ImputationPlotPayload` | `plot_imputation_auto_dashboard(...)`, `plot_imputation_method_dashboard(metrics, true_vals, pred_vals, method_name)`, candidate/article variants |
| `NormalizationPlotter` | `NormalizationPlotPayload` | `plot_normalization_dashboard()`, `plot_normalization_article_dashboard()` |

Dashboard methods return a patchwork composition or `None` when no meaningful composition is available. Lower-level panels may return Matplotlib figures/axes. Do not assume all plotting methods return the same type. Save a patchwork composition before constructing another dashboard because patchworklib uses shared figure state.

```python
from pimqc.plotting.normalization import NormalizationPlotter

plotter = NormalizationPlotter(normalization_result.audit.plot_payload)
dashboard = plotter.plot_normalization_dashboard()
if dashboard is not None:
    plotter.save_and_show_pw(
        pw_obj=dashboard,
        file_path="Normalization_Dashboard.svg",
        show_plot=False,
    )
```

`BasePlotter.save_and_show_pw()` provides vector export and notebook display for patchwork objects. `show_plot=False` suppresses display, not export. Display width is layout-dependent; compact 2 × 2 and single-brick layouts are not displayed at the same width as larger dashboards.

The six manuscript (`*_article_dashboard`) builders use the current 1.5.0
payloads and a shared **17.7 cm total export width**, including labels and
legends, when saved with `save_and_show_pw()`. Existing font families and
point sizes are retained; only panel geometry is adjusted. Article notebook
previews default to 60% width. Correction includes canonical D-ratio and
sample-structure diagnostics; normalization includes sample preservation.
These optional compact-layout APIs reuse computed audits rather than fitting
models again. They are not invoked by the public tutorial. Manuscript-specific
panel assembly, cropping and source figure exports are maintained privately
and are not required for package installation or scientific reproduction.

### 1.7.2 Rendering one assessment audit

`render_assessment(audit, output_dir, *, legend_mode="external", stage_label=None, include_dashboard=True, include_reference_charts=True, show_plot=False) -> None` is exported from `pimqc.plotting.assessment`.

```python
from pimqc.plotting.assessment import render_assessment

render_assessment(
    qa_result.audit,
    output_dir="qa_redraw",
    legend_mode="local",
    stage_label="Raw data",
    show_plot=False,
)
```

The renderer reuses stored correlation matrices, PCA/outlier results, RSD distributions, and reference diagnostics. A stage-label override applies to a detached context without mutating the audit. Skipped audits return without rendering; a non-skipped audit lacking its required payload raises `ValueError`.

### 1.7.3 Cross-stage comparison

`plot_assessment_comparison(audits, diagnostic, *, cols="auto", is_multi_batch=None)` constructs one diagnostic-wise comparison from a mapping of stage labels to assessment audits. Diagnostic names are `rsd`, `pca`, `correlation`, and `outlier`.

`render_assessment_comparison(audits, output_dir, *, cols="auto", is_multi_batch=None, show_plot=False) -> dict[str, Path]` exports all four grids. Mapping order is preserved; skipped assessments are excluded. An empty active mapping produces no assets.

These grids are drawn from saved audits, not by recalculating QA or stitching intermediate SVG panels. The standard single-stage QA dashboard and cross-stage shared-legend grids are distinct layouts.

### 1.7.4 Conditional layout behavior

When the outlier bar contains no actionable information, the single-stage QA dashboard uses four bricks: heatmap/PCA on the first row and RSD/outlier on the second. The heatmap body stays square, RSD shares its effective width, and the outlier legend is attached to the scatter's right side rather than placed in a standalone legend brick. Brick geometry and composition are implementation details, not scientific input parameters.

Filtering dashboards omit unavailable/skipped panels and adapt the flowchart to actual steps and group availability. Fixed-method imputation uses local density legends; Auto retains its shared comparison legend. Fixed-method normalization/correction does not fabricate candidate-selection panels. Preserve these distinctions when redrawing saved payloads.

## 1.8 Serialization API

### 1.8.1 Supported roots and disk format

The stable artifact boundary accepts three root families: `MetaboDataset`, typed `AuditPayload`, and typed `PlotPayload`. `StageResult`, `PipelineResult`, and live processors are not artifact roots. Save a stage's `.data` and `.audit` separately; for correction, save each selected dataset under its own destination.

Each artifact is a directory containing `manifest.json`, `payload.json`, and referenced member files. JSON stores structure/type metadata, JSONL stores table values, and NPY stores numerical arrays without pickle loading. This format does not replace ordinary stage CSV exports.

The manifest records format/schema identity, root payload, member paths, sizes, and SHA-256 checksums. Metadata alignment, pandas indexes, and typed payload identity are restored through registered wire types, not arbitrary class imports from an artifact.

### 1.8.2 Readers, writers, and validation

| Function | Return type |
| --- | --- |
| `write_metabo_dataset(dataset, path)` | Absolute artifact `Path` |
| `read_metabo_dataset(path, *, strict=True)` | `MetaboDataset` |
| `write_audit_payload(payload, path)` | Absolute artifact `Path` |
| `read_audit_payload(path, *, strict=True)` | Typed `AuditPayload` |
| `write_plot_payload(payload, path)` | Absolute artifact `Path` |
| `read_plot_payload(path, *, strict=True)` | Typed `PlotPayload` |
| `write_artifact(value, path)` | Generic writer for the three supported root families |
| `read_artifact(path, *, expected_kind=None, strict=True)` | Decoded root; optional kind is `dataset`, `audit`, or `plot_payload` |
| `validate_artifact(path, *, strict=True)` | `ArtifactManifest` after format/path/size/checksum validation |

Writer destinations must not already exist. Writes are staged and published as a directory; there is no overwrite switch. Choose a fresh path rather than deleting prior scientific results to rerun an example.

```python
from pimqc.serialization import (
    read_audit_payload,
    read_metabo_dataset,
    write_audit_payload,
    write_metabo_dataset,
)

# Both destination directories must be new.
write_metabo_dataset(normalization_result.data, "artifacts/normalized")
write_audit_payload(normalization_result.audit, "artifacts/normalization_audit")
restored_dataset = read_metabo_dataset("artifacts/normalized")
restored_audit = read_audit_payload("artifacts/normalization_audit")
```

Strict mode additionally rejects unlisted files. `strict=False` permits extra files but does not bypass checksums, required-member checks, registered-type checks, or format validation. `validate_artifact()` validates the container; reading also decodes and checks root identity against the manifest.

### 1.8.3 Errors

| Exception | Meaning |
| --- | --- |
| `SerializationError` | Base serialization error |
| `ArtifactExistsError` | Destination already exists |
| `UnsupportedSerializationTypeError` | Unsupported root or nested value type |
| `ArtifactValidationError` | Malformed/missing members, invalid paths, size/checksum mismatch, unexpected kind, or inconsistent identity |
| `UnsupportedFormatVersionError` | Unsupported artifact format version |
| `UnknownWireTypeError` | Unknown serialized type identifier |

These error classes and `ArtifactManifest`/`ArtifactFile` are exported from `pimqc.serialization`. Do not bypass validation failures by manually editing checksums or substituting default scientific values.

## 1.9 Reporting API

### 1.9.1 `ReportInput`

`ReportInput(pipeline_metrics, qa_metrics, metadata, resolved_config, asset_manifest={})` is a frozen dataclass with an independently created default asset mapping. `to_dict()` returns a JSON-friendly representation; `write_json(path) -> Path` persists it. There is no automatic `from_stage_results()` constructor.

| Field | Source |
| --- | --- |
| `pipeline_metrics` | Processing audits, in the stage-key structure shown in `pipeline.py` |
| `qa_metrics` | Assessment audits, including separate selected correction checkpoints |
| `metadata` | Run metadata such as date, package version, acquisition mode, and multi-batch flag |
| `resolved_config` | Validated configuration used for the run; explicit per-stage changes must also be represented accurately in custom workflows |
| `asset_manifest` | Logical asset names mapped to report-relative paths, normally returned by `VisualAssetReporter` |

Use `pipeline_result.report_input` for the native pipeline. Custom workflows should build metrics from actual audits and retain the same report context structure; arbitrary dictionaries or a live `StageResult` do not supply the required narrative evidence. Missing evidence is not an invitation to insert numerical defaults.

### 1.9.2 `VisualAssetReporter`

`VisualAssetReporter(base_dir)` establishes the output workspace. `compile_assessor_report(audits, is_multi_batch=True, report_folder="13_Report_Markdown", cols="auto", show_plot=True) -> dict[str, str]` renders the four audit-backed QA grids and returns paths relative to the report directory.

Pass the same explicit report folder to both reporters: their low-level defaults differ, while the pipeline uses `07_Report_Summary`. Supply named `AssessQualityAuditPayload` objects, not processor instances or SVG paths.

### 1.9.3 Narrative and document export

`NarrativeStatsReporter(base_dir)` reads packaged templates. Its public workflow is `generate_markdown(report_input, report_folder="08_Report_Summary") -> None`, followed optionally by `export_report(pdf_engine="weasyprint") -> bool`. `consolidate_metrics(report_input)` builds the template context from supplied evidence.

```python
from pimqc.reporting import NarrativeStatsReporter

# Use the same root that already contains this run's stage figures.
reporter = NarrativeStatsReporter(base_dir="outputs")
reporter.generate_markdown(
    pipeline_result.report_input,
    report_folder="07_Report_Summary",
)
```

For complete figure references, this example requires a pipeline result originally produced with `output_dir="outputs"`, or a custom `ReportInput` backed by equivalent stage/QA assets. A computation-only result has metrics but no generated assets; writing its narrative alone does not create those figures.

Markdown generation writes `Report_Comprehensive.md`, `Report_Brief.md`, and `Report_Input.json`. Document export checks image references and unresolved template content before conversion; this validation belongs to the narrative reporter, not the visual reporter. Stage reports reference `MV_Classification_Dashboard.svg` where that dashboard is available.

PDF/HTML export uses explicitly provisioned tools such as Pandoc, WeasyPrint, XeLaTeX, and rsvg-convert, with fallback paths. It never downloads or installs dependencies. Missing tools are reported; Markdown and scientific stage outputs remain available. Check the boolean export result. The CLI exits with `0` on success, `1` on input/scientific execution failure, and `2` when scientific processing finishes but report export fails; quiet mode still emits errors. A successful HTML fallback counts as successful report export, not as proof that a PDF exists.

## 1.10 Integration boundaries

The 1.5.0 API retains the composition-based redesign introduced in 1.4.0. New workflows should exchange `MetaboDataset`, `StageResult`, typed audits, and detached plot payloads. Do not recreate the removed `MetaboInt` pandas-subclass contract, attach cross-stage state implicitly to `DataFrame.attrs`, or assume former result dictionaries are accepted.

Some current implementation helpers and the exported `FeatureFilter` facade still exist in source. They are not evidence that pre-1.4 workflows remain compatible; the independent filters and named run entry points above are the recommended application interfaces. Private algorithm workers, caches, and runner internals are not an integration contract.

The scientific core has no Rachis, QIIME 2, or `q2-types` dependency. Framework registration, semantic types, transformers, and plugin pipelines belong in the separate adapter project. Adapters should consume native contracts and serialized artifacts without requiring changes to the core dataset model.
