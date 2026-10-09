# Optional R backend — pi-metaboqc 1.5.0

This guide covers installation, source requirements and runtime behavior for the optional R backend. Return to the [README](../README.md) for the Python-only workflow.

R is not needed for the default Python implementations. Version 1.5.0 adds the following explicit original-package paths; Quantile remains Python-only and unchanged.

The `[r]` extra installs the Python bridge (`rpy2`), not R itself or the upstream R packages. Install the packages needed by your selected methods; the commands below cover all supported providers.

| Stage | Explicit method settings with `implementation="r"` | Original R call |
| --- | --- | --- |
| Imputation | `r_route_method="BPCA"`, `s_route_method="QRILC"` | `pcaMethods::pca` / `completeObs`; `imputeLCMD::impute.QRILC` |
| Normalization | `norm_method="VSN"` | `vsn::vsn2` / `vsn::predict` |
| Correction | `base_est="QC-RLSC", rlsc_robust=True` | `stats::loess` with the MetaNorm/fANCOVA GCV fitting design |
| Correction, reference worker | `base_est="Metanorm-rLOESS"` | Exported `metanorm::metanormWorker` |
| Correction | `base_est="WaveICA 2.0"` | `WaveICA2.0::WaveICA_2.0` |
| Correction | `base_est="RUV-III"`, explicit `ruv_control_features` | `ruv::RUVIII` |
| Correction | `base_est="SERRF"`, explicit `serrf_r_source` | Author's `serrfR` extracted from pinned Shiny-SERRF source; `ranger` engine |

`R-route` means **reconstruction route**, not execution in the R language; `S-route` means **special-handling route** for features retained by group-pattern or low-intensity QC rescue. Both routes are available with either implementation backend. The preferred input options are `r_route_method` and `s_route_method`; historical `mar_method` and `mnar_method` remain accepted and are retained in serialized configuration for compatibility. Select the language backend independently with `implementation`.

`implementation="r"` also supports `AUTO`: existing candidate sets and stage-specific scoring are retained, with R replacing robust QC-RLSC/SERRF/RUV-III/WaveICA in correction, BPCA/QRILC in imputation, and VSN in normalization. Candidates without an R adapter remain explicitly Python; a failed R candidate is recorded as failed, never retried with its Python counterpart. The audit identifies each candidate's actual backend and the selected backend. The direct `Metanorm-rLOESS` reference worker remains outside AUTO; it is not added as a duplicate candidate. `implementation` is accepted by each processor constructor, its `run_*` call, or the corresponding TOML/JSON section; it is unrelated to Joblib's `serrf_backend` and `regression_backend` settings. See the [native API reference](https://github.com/PHOENIXcenter/pi-metaboqc/blob/main/docs/native_api.md#135-optional-r-implementations) for examples and input constraints.

Robust QC-RLSC now follows MetaNorm's rLOESS design: log2 intensities, quadratic symmetric LOESS, continuous GCV span selection, and mean-restoring residual correction. The default Python implementation uses scikit-misc's compiled LOESS kernel without R; the optional R implementation uses the original R fitting functions. Both refit within QC validation folds. Unsupported extrapolation and failed fits remain unavailable, with coverage reported; full-fit predictions never replace held-out QC predictions. This changes the former raw-scale robust QC-RLSC algorithm, so results from earlier releases must not be presented as current-version validation.

## 1. Install R and rpy2

The development machine used for this release runs **Windows, Python 3.13.13, R 4.5.2 (64-bit), and rpy2 3.6.7**. These are tested example versions, not a claim that every OS or package combination has been validated.

On Windows, install [R from CRAN](https://cran.r-project.org/bin/windows/base/). The local example installation is `D:\R\R-4.5.2`; replace it with your own path. Set `R_HOME` before starting Python or the notebook kernel if automatic registry discovery selects the wrong installation:

```powershell
$env:R_HOME = "D:\R\R-4.5.2"
$env:R_ARCH = "/x64"
$env:PATH = "$env:R_HOME\bin\x64;$env:R_HOME\bin;$env:PATH"
```

On Linux/macOS, an alternative is Conda-managed R and rpy2:

```bash
conda install -c conda-forge r-base=4.5.2 rpy2 -y
```

Then, from the 1.5.0 source checkout, install the Python package with its R extra:

```bash
pip install -e ".[r]"
```

For a published 1.5.0 distribution, the equivalent release installation is `pip install "pi-metaboqc[r]==1.5.0"`. If pip needs to build rpy2 or an R package from source, the corresponding compiler and R development tools may be required; on Windows use the [Rtools version matching R](https://cran.r-project.org/bin/windows/Rtools/). A working Python-only install does not establish that embedded R is configured correctly.

## 2. Install the original R packages

Run these commands in the **same R installation** selected by rpy2:

```r
install.packages(c("BiocManager", "remotes"),
                 repos = "https://cloud.r-project.org")
BiocManager::install(c("vsn", "pcaMethods", "imputeLCMD"),
                     ask = FALSE, update = FALSE)
remotes::install_github(
    "UGent-LIMET/Metanorm@ef6f6ee66436b8222f92d926e829aaa959fc56da",
    upgrade = "never"
)
install.packages(c("ruv", "ranger", "waveslim", "ica", "JADE", "corpcor"),
                 repos = "https://cloud.r-project.org")
remotes::install_github(
    "dengkuistat/WaveICA_2.0@56fff2e6b0410b5957c6ea83bc658241df222f41",
    upgrade = "never"
)
```

The verified local packages are **vsn 3.78.1, pcaMethods 2.2.0, imputeLCMD 2.1, metanorm 0.10.2, WaveICA2.0 0.1.0, ruv 0.9.7.2, and ranger 0.18.0**. The GitHub commits above identify the installed Metanorm and WaveICA sources. Bioconductor resolves versions compatible with your R release; those installation commands are not a lockfile for the complete local environment. pi-metaboqc does not install R packages or modify package libraries automatically.

## 3. Diagnose and test the optional backend

```bash
Rscript --version
python -m rpy2.situation
python -c "from pimqc.processing.r_backend import check_r_environment; print(check_r_environment(('vsn', 'pcaMethods', 'imputeLCMD', 'metanorm', 'WaveICA2.0', 'ruv', 'ranger')))"
python -m pytest tests/reference -q
```

Run tests from the source checkout with `pip install -e ".[test,r]"`. Check that the diagnostic reports `available: True`, the expected `r_home`, and a version for every required package. A skipped optional test is not a successful validation of that method. `python -m pytest tests/reference -q` checks the production R adapters and robust LOESS contracts; default `python -m pytest -q` does not initialize R. Native development tests need only `pip install -e ".[test]"`. If Windows refuses its default pytest temporary directory, pass `--basetemp` with a dedicated scratch directory; pytest may clear its contents.

R execution runs serially on the main Python thread; `n_jobs` does not parallelize embedded R. Restart the Python process or notebook kernel after changing R installations or environment variables. The stage audit records the actual R function, package/version, available source commit, parameters, random seed, warnings, and scale transformations. Saved datasets, audits, plots, and reports do not require R to be read. Passing adapter tests verifies calls to the installed upstream implementation, not equivalence of every Python port or scientific superiority on a new dataset.

## 4. Prepare the optional local SERRF source

[Shiny-SERRF](https://github.com/slfan2013/Shiny-SERRF) is an author-provided application, not an installable R package. Its repository does not declare an explicit license, so pi-metaboqc does not bundle its source. Obtain the pinned [app.R](https://raw.githubusercontent.com/slfan2013/Shiny-SERRF/fb19f2a80a742dcdd469a9d9bcc1ecef8ee585a2/app.R) separately, subject to the upstream usage terms; confirm permission before redistribution. The accepted original file has SHA-256 `3e37953afa70edfc4310afebcd58e6cac83f64cc2cbc9686bcf84ae4a87b8551`.

Pass its local path as `serrf_r_source`, or use `pimqc.processing.correction.serrf_r.prepare_serrf_source(source_path, output_path)` to produce a reusable, verified standalone function file. Extraction parses R syntax without running the application, evaluating top-level code, installing packages, or starting Shiny. The adapter isolates UI progress callbacks and applies recorded runtime NA-indexing and per-forest seed adaptations to the author's function; the verified source file is not modified. Only the pinned original or its verified extraction is accepted; no runtime download occurs. For optional real-source tests, set `PIMQC_SERRF_R_SOURCE` to the saved `app.R` path before running pytest.

The upstream `serrfR` body calls `set.seed(1)` internally for each forest. At execution, the adapter verifies that the extracted AST contains exactly that one seed call and replaces only its constant with `random_state`, for both full fit and held-out-QC fits. Thus `random_state=123` gives an effective per-forest seed of 123 as well as the outer R seed. Provenance distinguishes the upstream `original_model_seed=1` from `effective_model_seed` and records `runtime_seed_adaptation`; the host RNG is restored afterward. Source bytes, extraction checksums, tree count, predictors, and other algorithm settings remain unchanged. This is a runtime seed-adapted source function, not an unmodified original-function execution. Subsequent upstream random repairs can also be affected by the changed RNG trajectory; their replacement distributions are unchanged.

The original SERRF zero/negative-value repair uses logical subscripts that can contain NA and fail when several values are missing. The full-fit and held-out-QC adapters share an NA-safe subscript repair without changing the random replacement distributions. A feature with an entirely missing fitting batch has no batch minimum for these replacements: it is excluded from that fit, returned as unavailable across non-Blank samples, and recorded explicitly rather than borrowing another batch's values. Fold eligibility uses only training QCs and real biological samples. Complete supported inputs retain the original numerical path apart from the explicit seed adaptation; this remains a documented adaptation, not a claim that the upstream code handles arbitrary missingness.

Original WaveICA/RUV-III kernels reject missing values, so their Python/R adapters now share temporary feature-median completion based only on observed non-Blank samples: raw space for WaveICA and log space for RUV. Original missing positions are restored after correction, with counts and scale recorded in the audit. This is an explicit input adaptation, not native missing-data support or formal imputation; all-missing features and insufficient real RUV control support remain errors. Original SERRF instead receives NaN and zero inputs unchanged and performs its own stochastic repairs; its adapter restores original non-Blank NA positions in full-fit and OOF outputs before the common correction-domain policy, leaving these cells for the later imputation stage. Input zeros are not additionally restored as NA. Temporary values can still influence fitted corrections even after original NA positions are restored. Upstream failures remain errors; arbitrary missingness patterns are not guaranteed to work. No pipeline reorder occurs. In AUTO, failed candidates remain visible. Blank columns remain unchanged and are excluded from these three R fits. RUV-III requires declared negative controls and uses pooled QCs as technical replicates by default; a metadata column can supply an explicit technical-repeat design. RUV-III is not RUV-III-C.

R SERRF adds a pi-metaboqc QC-validation wrapper: refit the original function per fold, capture its fitted ranger models and training scales, and predict excluded QCs without using them in fitting or alignment. This is an adapter extension, not an upstream OOF interface. `SignalCorrector` requires `cv_folds >= 2`; its built-in default is 5, while the bundled demo configuration explicitly uses 3. Only the internal `SERRFRCorrector` adapter accepts `cv_folds=0` for a full-fit reference call; this is not a valid TOML or stage setting. Python and R share QC fold assignments. RUV-III and WaveICA retain descriptive full-data AUTO evaluation. See the [correction API](native_api.md#147-signalcorrector) for method-specific scoring fallback and eligibility rules, and [original R correction adapters](native_api.md#1471-original-r-correction-adapters) for parameter mappings.
