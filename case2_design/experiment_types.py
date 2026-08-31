from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Protocol, Tuple

import numpy as np


class Action(str, Enum):
    A1 = "A1"
    A2 = "A2"
    A3 = "A3"
    A4 = "A4"


class Phase(str, Enum):
    FEASIBILITY = "feasibility"
    COST = "cost"


class SelectorErrorKind(str, Enum):
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_ERROR = "provider_error"
    INVALID_JSON = "invalid_json"
    SCHEMA_ERROR = "schema_error"
    INVALID_ACTION = "invalid_action"
    SELECTOR_INTERNAL_ERROR = "selector_internal_error"


@dataclass(frozen=True)
class SelectionDecision:
    requested_action: Optional[Action]
    applied_action: Action
    fallback_used: bool = False
    error_kind: Optional[SelectorErrorKind] = None
    elapsed_seconds: float = 0.0
    llm_call_id: Optional[str] = None


@dataclass(frozen=True)
class SearchConfig:
    hours: int = 24
    population_size: int = 600
    max_iterations: int = 500
    distance: float = 240.0
    speed_max: float = 11.0
    velocity_clamp_v: float = 2.2
    velocity_clamp_qe: float = 0.2
    velocity_clamp_qg: float = 0.2
    omega_max: float = 0.9
    omega_min: float = 0.4
    c_max: float = 1.5
    c_min: float = 0.5
    alpha: float = 0.8
    zeta: float = 0.8
    c3: float = -0.4
    penalty_lambda: float = 1.0e6
    feasibility_tolerance: float = 1.0e-10
    numerical_epsilon: float = 1.0e-12
    decision_interval: int = 20

    @property
    def dimensions(self):  # type: () -> int
        return 3 * self.hours

    @property
    def search_budget(self):  # type: () -> int
        return self.population_size * (self.max_iterations + 1)


@dataclass(frozen=True)
class ActionParameters:
    a2_target_fraction: float = 0.25
    a2_differential_weight: float = 0.5
    a2_crossover_probability: float = 0.9
    a3_target_fraction: float = 0.15
    a3_sigma_0: float = 0.05
    a3_sigma_min: float = 0.005
    a4_target_fraction: float = 0.25
    a4_beta_min: float = 0.2
    a4_beta_max: float = 0.8


@dataclass
class RandomStreams:
    core: np.random.Generator
    a2: np.random.Generator
    a3: np.random.Generator
    a4: np.random.Generator
    selector: np.random.Generator

    def for_action(self, action):  # type: (Action) -> np.random.Generator
        if action == Action.A2:
            return self.a2
        if action == Action.A3:
            return self.a3
        if action == Action.A4:
            return self.a4
        return self.a2


@dataclass
class SearchState:
    iteration: int
    search_nfe: int
    positions: np.ndarray
    velocities: np.ndarray
    previous_positions: np.ndarray
    previous_velocities: np.ndarray
    evaluation: Dict[str, np.ndarray]
    personal_best_positions: np.ndarray
    personal_best_cost: np.ndarray
    personal_best_cv: np.ndarray
    personal_best_cv_components: np.ndarray
    global_best_position: np.ndarray
    global_best_cost: float
    global_best_cv: float
    global_best_cv_components: Tuple[float, float, float, float]
    first_feasible_nfe: Optional[int]
    active_action: Optional[Action]
    stagnation_iterations: int


@dataclass(frozen=True)
class SelectorSummary:
    window_index: int
    iteration: int
    search_nfe: int
    budget_fraction: float
    phase: Phase
    global_best_cost: float
    global_best_cv: float
    cv_r: float
    cv_soc: float
    cv_e: float
    cv_d: float
    feasible_fraction: float
    population_cv_q25: float
    population_cv_median: float
    population_cv_q75: float
    normalized_diversity: float
    stagnation_iterations: int
    previous_action: Optional[Action]


@dataclass(frozen=True)
class StrategyOutcome:
    window_index: int
    phase_at_start: Phase
    phase_at_end: Phase
    start_iteration: int
    end_iteration: int
    start_nfe: int
    end_nfe: int
    start_best_cost: float
    end_best_cost: float
    start_best_cv: float
    end_best_cv: float
    first_feasible_appeared: bool
    first_feasible_nfe: Optional[int]
    relative_improvement: float
    requested_action: Optional[Action]
    applied_action: Action
    fallback_used: bool
    error_kind: Optional[SelectorErrorKind]
    elapsed_seconds: float
    llm_call_id: Optional[str]


class SelectorProtocol(Protocol):
    def select(
        self, summary: SelectorSummary, rng: np.random.Generator
    ) -> SelectionDecision:
        ...

    def observe(self, action: Action, outcome: StrategyOutcome) -> None:
        ...


@dataclass(frozen=True)
class SearchRun:
    best_position: np.ndarray
    audit_evaluation: Dict[str, np.ndarray]
    history: List[Dict[str, Any]]
    outcomes: List[StrategyOutcome]
    search_nfe: int
    audit_nfe: int
    first_feasible_nfe: Optional[int]


Evaluation = Mapping[str, np.ndarray]
