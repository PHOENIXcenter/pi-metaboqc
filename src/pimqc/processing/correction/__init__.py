"""Signal-correction algorithms and orchestration exports.

The package exposes correction engines required by the public pipeline and by
method-level tests, while low-level QC-RLSC helpers remain available for
reference validation and specialized analysis workflows.
"""

from .algorithms import _numba_loess_robust, _select_loess_span_oof
from .analysis import SignalCorrector
from .metanorm import MetanormRLOESSCorrector
from .regression import RegressionCorrector
from .rloess import RobustRLSCCorrector
from .ruv import RUVCorrector
from .ruv_r import RUVIIIRCorrector
from .serrf import SERRFCorrector
from .serrf_r import SERRFRCorrector, prepare_serrf_source
from .waveica import WaveICA2Corrector
from .waveica_r import WaveICARCorrector

__all__ = [
    "SignalCorrector",
    "MetanormRLOESSCorrector",
    "RegressionCorrector",
    "RobustRLSCCorrector",
    "RUVCorrector",
    "RUVIIIRCorrector",
    "SERRFCorrector",
    "SERRFRCorrector",
    "prepare_serrf_source",
    "WaveICA2Corrector",
    "WaveICARCorrector",
    "_numba_loess_robust",
    "_select_loess_span_oof",
]
