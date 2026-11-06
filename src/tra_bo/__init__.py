"""TRA-BO: Bayesian optimization with Temporal Rank Agreement."""

from .llm import (
    BOState,
    FixedAcquisitionSelector,
    LLMConfig,
    LLMAcquisitionSelector,
    OpenAICompatibleClient,
)
from .evaluation import (
    incumbent_future_loss_trajectory,
    normalized_final_regret,
    normalized_future_regret_auc,
)
from .optimizer import OptimizationResult, TRABO, TRABOConfig, TrialRecord
from .splits import ForecastWindow, OrderedTemporalSplit, make_ordered_split
from .spaces import decode_configuration, generate_sobol_pool, space_dimension
from .temporal import TRAEstimate, compute_tra, summarize_windows, temporal_target

__all__ = [
    "BOState",
    "FixedAcquisitionSelector",
    "ForecastWindow",
    "LLMConfig",
    "LLMAcquisitionSelector",
    "OpenAICompatibleClient",
    "OrderedTemporalSplit",
    "OptimizationResult",
    "TRAEstimate",
    "TRABO",
    "TRABOConfig",
    "TrialRecord",
    "compute_tra",
    "decode_configuration",
    "generate_sobol_pool",
    "incumbent_future_loss_trajectory",
    "make_ordered_split",
    "normalized_final_regret",
    "normalized_future_regret_auc",
    "space_dimension",
    "summarize_windows",
    "temporal_target",
]
