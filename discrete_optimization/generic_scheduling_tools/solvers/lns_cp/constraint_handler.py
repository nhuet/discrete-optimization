from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Generic, Iterable, Optional

from discrete_optimization.generic_scheduling_tools.allocation import (
    AllocationCpSolver,
    AllocationProblem,
    AllocationSolution,
)
from discrete_optimization.generic_scheduling_tools.base import (
    Task,
    TasksCpSolver,
    TasksProblem,
    TasksSolution,
)
from discrete_optimization.generic_scheduling_tools.enums import StartOrEnd
from discrete_optimization.generic_scheduling_tools.multimode import (
    MultimodeCpSolver,
    MultimodeProblem,
    MultimodeSolution,
)
from discrete_optimization.generic_scheduling_tools.scheduling import (
    SchedulingCpSolver,
    SchedulingProblem,
    SchedulingSolution,
)
from discrete_optimization.generic_scheduling_tools.solvers.lns_cp.neighbor_tools import (
    NeighborBuilder,
    build_default_neighbor_builder,
)
from discrete_optimization.generic_tools.cp_tools import SignEnum
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    FloatHyperparameter,
    IntegerHyperparameter,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparametrizable import (
    Hyperparametrizable,
)
from discrete_optimization.generic_tools.lns_tools import ConstraintHandler
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

logger = logging.getLogger(__name__)


class ParamsConstraintBuilder(Hyperparametrizable):
    hyperparameters = [
        IntegerHyperparameter(name="minus_delta_primary", low=0, default=100, high=200),
        IntegerHyperparameter(name="plus_delta_primary", low=0, default=100, high=200),
        IntegerHyperparameter(name="minus_delta_secondary", low=0, default=0, high=10),
        IntegerHyperparameter(name="plus_delta_secondary", low=0, default=0, high=10),
        IntegerHyperparameter(
            name="minus_delta_primary_duration", default=5, low=0, high=10
        ),
        IntegerHyperparameter(
            name="plus_delta_primary_duration", default=5, low=0, high=10
        ),
        IntegerHyperparameter(
            name="minus_delta_secondary_duration", default=5, low=0, high=10
        ),
        IntegerHyperparameter(
            name="plus_delta_secondary_duration", default=5, low=0, high=10
        ),
        CategoricalHyperparameter(
            name="constraint_max_time_to_current_solution",
            choices=[True, False],
            default=False,
        ),
        FloatHyperparameter(
            name="fraction_of_task_assigned_multiskill", default=0.6, low=0.0, high=1.0
        ),
        CategoricalHyperparameter(
            name="except_assigned_multiskill_primary_set",
            choices=[True, False],
            default=False,
            depends_on=("first_method_multiskill", [True]),
        ),
        CategoricalHyperparameter(
            name="first_method_multiskill", choices=[True, False], default=True
        ),
        CategoricalHyperparameter(
            name="second_method_multiskill", choices=[True, False], default=False
        ),
        CategoricalHyperparameter(
            name="additional_methods", choices=[True, False], default=False
        ),
    ]

    def __init__(
        self,
        minus_delta_primary: int,
        plus_delta_primary: int,
        minus_delta_secondary: int,
        plus_delta_secondary: int,
        minus_delta_primary_duration: int = 5,
        plus_delta_primary_duration: int = 5,
        minus_delta_secondary_duration: int = 5,
        plus_delta_secondary_duration: int = 5,
        constraint_max_time_to_current_solution: bool = False,
        margin_constraint_max_time_to_current_solution: int = 20,
        fraction_of_task_assigned_multiskill: float = 0.6,
        except_assigned_multiskill_primary_set: bool = False,
        first_method_multiskill: bool = True,
        second_method_multiskill: bool = False,
        additional_methods: bool = False,
    ):
        self.margin_constraint_max_time_to_current_solution = (
            margin_constraint_max_time_to_current_solution
        )
        self.minus_delta_primary = minus_delta_primary
        self.plus_delta_primary = plus_delta_primary
        self.minus_delta_secondary = minus_delta_secondary
        self.plus_delta_secondary = plus_delta_secondary
        self.minus_delta_primary_duration = minus_delta_primary_duration
        self.plus_delta_primary_duration = plus_delta_primary_duration
        self.minus_delta_secondary_duration = minus_delta_secondary_duration
        self.plus_delta_secondary_duration = plus_delta_secondary_duration
        self.constraint_max_time_to_current_solution = (
            constraint_max_time_to_current_solution
        )
        self.fraction_of_task_assigned_multiskill = fraction_of_task_assigned_multiskill
        self.except_assigned_multiskill_primary_set = (
            except_assigned_multiskill_primary_set
        )
        self.first_method_multiskill = first_method_multiskill
        self.second_method_multiskill = second_method_multiskill
        self.additional_methods = additional_methods

    @staticmethod
    def default():
        return ParamsConstraintBuilder(
            minus_delta_primary=5,
            plus_delta_primary=5,
            minus_delta_secondary=0,
            plus_delta_secondary=0,
        )


class BaseConstraintExtractor(ABC, Generic[Task]):
    """Base class for constraint extractor.

    The constraints are extracted from a current solution + tasks subset.

    """

    @abstractmethod
    def add_constraints(
        self,
        current_solution: TasksSolution[Task],
        solver: TasksCpSolver[Task],
        tasks_primary: set[Task],
        tasks_secondary: set[Task],
        params_constraint_builder: ParamsConstraintBuilder = None,
        **kwargs: Any,
    ) -> list[Any]:
        """Extract constraints and add them to the cp model.

        Args:
            current_solution:
            solver:
            tasks_primary:
            tasks_secondary:
            params_constraint_builder:
            **kwargs:

        Returns:

        """
        ...


class ConstraintExtractorList(BaseConstraintExtractor[Task]):
    """Extractor adding constraints from multiple sub-extractors."""

    def __init__(
        self,
        extractors: list[BaseConstraintExtractor[Task]],
    ):
        self.extractors = extractors

    def add_constraints(
        self,
        current_solution: TasksSolution[Task],
        solver: TasksCpSolver[Task],
        tasks_primary: set[Task],
        tasks_secondary: set[Task],
        params_constraint_builder: ParamsConstraintBuilder = None,
        **kwargs: Any,
    ) -> list[Any]:
        constraints = []
        for extractor in self.extractors:
            constraints += extractor.add_constraints(
                current_solution=current_solution,
                solver=solver,
                tasks_primary=tasks_primary,
                tasks_secondary=tasks_secondary,
                params_constraint_builder=params_constraint_builder,
                **kwargs,
            )
        return constraints


class SchedulingConstraintExtractor(BaseConstraintExtractor[Task]):
    def __init__(
        self,
        params_constraint_builder: ParamsConstraintBuilder = None,
    ):

        if params_constraint_builder is None:
            self.params_constraint_builder = ParamsConstraintBuilder.default()
        else:
            self.params_constraint_builder = params_constraint_builder

    def add_constraints(
        self,
        current_solution: TasksSolution[Task],
        solver: TasksCpSolver[Task],
        tasks_primary: set[Task],
        tasks_secondary: set[Task],
        params_constraint_builder: ParamsConstraintBuilder = None,
        **kwargs: Any,
    ) -> list[Any]:
        if not (
            isinstance(current_solution, SchedulingSolution)
            and isinstance(solver, SchedulingCpSolver)
        ):
            raise ValueError(
                f"{self.__class__.__name__} extract constraints only "
                f"if solution and solver are related to a scheduling problem."
            )

        if params_constraint_builder is None:
            params_constraint_builder = self.params_constraint_builder
        max_time = current_solution.get_max_end_time()
        constraints = []
        for task in tasks_primary:
            start_time_j = current_solution.get_start_time(task)
            constraints += solver.add_constraint_on_task(
                task=task,
                start_or_end=StartOrEnd.START,
                sign=SignEnum.UEQ,
                time=max(
                    0, start_time_j - params_constraint_builder.minus_delta_primary
                ),
            )
            constraints += solver.add_constraint_on_task(
                task=task,
                start_or_end=StartOrEnd.START,
                sign=SignEnum.LEQ,
                time=min(
                    max_time,
                    start_time_j + params_constraint_builder.plus_delta_primary,
                )
                if params_constraint_builder.constraint_max_time_to_current_solution
                else start_time_j + params_constraint_builder.plus_delta_primary,
            )
        for task in tasks_secondary:
            if task in tasks_primary:
                continue
            start_time_j = current_solution.get_start_time(task)
            if (
                params_constraint_builder.minus_delta_secondary == 0
                and params_constraint_builder.plus_delta_secondary == 0
            ):
                constraints += solver.add_constraint_on_task(
                    task=task,
                    start_or_end=StartOrEnd.START,
                    sign=SignEnum.EQUAL,
                    time=start_time_j,
                )
            else:
                constraints += solver.add_constraint_on_task(
                    task=task,
                    start_or_end=StartOrEnd.START,
                    sign=SignEnum.UEQ,
                    time=max(
                        0,
                        start_time_j - params_constraint_builder.minus_delta_secondary,
                    ),
                )
                constraints += solver.add_constraint_on_task(
                    task=task,
                    start_or_end=StartOrEnd.START,
                    sign=SignEnum.LEQ,
                    time=min(
                        max_time,
                        start_time_j + params_constraint_builder.plus_delta_secondary,
                    )
                    if params_constraint_builder.constraint_max_time_to_current_solution
                    else start_time_j + params_constraint_builder.plus_delta_secondary,
                )
        if params_constraint_builder.constraint_max_time_to_current_solution:
            margin = 0
        else:
            margin = (
                params_constraint_builder.margin_constraint_max_time_to_current_solution
            )
        constraints += solver.add_constraint_on_max_end_time(
            sign=SignEnum.LEQ,
            time=max_time + margin,
        )
        return constraints


class ChainingConstraintExtractor(BaseConstraintExtractor[Task]):
    def __init__(
        self,
        frac_fixed_chaining: float = 0.25,
    ):
        self.frac_fixed_chaining = frac_fixed_chaining

    def add_constraints(
        self,
        current_solution: TasksSolution[Task],
        solver: TasksCpSolver[Task],
        tasks_primary: set[Task],
        tasks_secondary: set[Task],
        params_constraint_builder: ParamsConstraintBuilder = None,
        **kwargs: Any,
    ) -> list[Any]:
        if not (
            isinstance(current_solution, SchedulingSolution)
            and isinstance(solver, SchedulingCpSolver)
        ):
            raise ValueError(
                f"{self.__class__.__name__} extract constraints only "
                f"if solution and solver are related to a scheduling problem."
            )

        all_tasks = current_solution.problem.tasks_list
        tasks = random.sample(all_tasks, int(self.frac_fixed_chaining * len(all_tasks)))
        constraints = []
        for task1 in tasks:
            for task2 in tasks:
                if current_solution.get_end_time(
                    task1
                ) == current_solution.get_start_time(task2):
                    constraints += solver.add_constraint_chaining_tasks(
                        task1=task1, task2=task2
                    )
        return constraints


class MultimodeConstraintExtractor(BaseConstraintExtractor[Task]):
    """Extractor adding constraints on modes."""

    def __init__(
        self,
        fix_primary_tasks_modes: bool = False,
        fix_secondary_tasks_modes: bool = True,
    ):
        self.fix_primary_tasks_modes = fix_primary_tasks_modes
        self.fix_secondary_tasks_modes = fix_secondary_tasks_modes

    def add_constraints(
        self,
        current_solution: TasksSolution[Task],
        solver: TasksCpSolver[Task],
        tasks_primary: set[Task],
        tasks_secondary: set[Task],
        params_constraint_builder: ParamsConstraintBuilder = None,
        **kwargs: Any,
    ) -> list[Any]:
        if not (
            isinstance(current_solution, MultimodeSolution)
            and isinstance(solver, MultimodeCpSolver)
        ):
            raise ValueError("current_solution and solver must manage tasks modes.")

        constraints = []
        if self.fix_primary_tasks_modes:
            for task in tasks_primary:
                constraints += solver.add_constraint_on_task_mode(
                    task=task, mode=current_solution.get_mode(task)
                )
        if self.fix_secondary_tasks_modes:
            for task in tasks_secondary:
                constraints += solver.add_constraint_on_task_mode(
                    task=task, mode=current_solution.get_mode(task)
                )
        return constraints


class SubtasksAllocationConstraintExtractor(BaseConstraintExtractor[Task]):
    def __init__(
        self,
        fix_secondary_tasks_modes: bool = False,
        frac_random_fixed_tasks: float = 0.6,
    ):
        self.frac_random_fixed_tasks = frac_random_fixed_tasks
        self.fix_secondary_tasks_modes = fix_secondary_tasks_modes

    def add_constraints(
        self,
        current_solution: TasksSolution[Task],
        solver: TasksCpSolver[Task],
        tasks_primary: set[Task],
        tasks_secondary: set[Task],
        params_constraint_builder: ParamsConstraintBuilder = None,
        **kwargs: Any,
    ) -> list[Any]:
        if not (
            isinstance(current_solution, AllocationSolution)
            and isinstance(solver, AllocationCpSolver)
        ):
            raise ValueError(
                "current_solution and solver must manage resource allocation."
            )
        if self.fix_secondary_tasks_modes:
            tasks = tasks_secondary
        else:
            all_tasks = current_solution.problem.tasks_list
            tasks = set(
                random.sample(
                    all_tasks,
                    int(self.frac_random_fixed_tasks * len(all_tasks)),
                )
            )
        return solver.add_constraint_same_allocation_as_ref(
            ref=current_solution, tasks=tasks
        )


class SubresourcesAllocationConstraintExtractor(BaseConstraintExtractor[Task]):
    def __init__(
        self,
        frac_random_fixed_unary_resources: float = 0.5,
    ):
        self.frac_random_fixed_unary_resources = frac_random_fixed_unary_resources

    def add_constraints(
        self,
        current_solution: TasksSolution[Task],
        solver: TasksCpSolver[Task],
        tasks_primary: set[Task],
        tasks_secondary: set[Task],
        params_constraint_builder: ParamsConstraintBuilder = None,
        **kwargs: Any,
    ) -> list[Any]:
        if not (
            isinstance(current_solution, AllocationSolution)
            and isinstance(solver, AllocationCpSolver)
        ):
            raise ValueError(
                "current_solution and solver must manage resource allocation."
            )
        all_unary_resources = current_solution.problem.unary_resources_list
        unary_resources = set(
            random.sample(
                all_unary_resources,
                int(self.frac_random_fixed_unary_resources * len(all_unary_resources)),
            )
        )
        return solver.add_constraint_same_allocation_as_ref(
            ref=current_solution, unary_resources=unary_resources
        )


class NbChangesAllocationConstraintExtractor(BaseConstraintExtractor[Task]):
    def __init__(
        self,
        nb_changes: int = 10,
    ):
        self.nb_changes = nb_changes

    def add_constraints(
        self,
        current_solution: TasksSolution[Task],
        solver: TasksCpSolver[Task],
        tasks_primary: set[Task],
        tasks_secondary: set[Task],
        params_constraint_builder: ParamsConstraintBuilder = None,
        **kwargs: Any,
    ) -> list[Any]:
        if not (
            isinstance(current_solution, AllocationSolution)
            and isinstance(solver, AllocationCpSolver)
        ):
            raise ValueError(
                "current_solution and solver must manage resource allocation."
            )
        return solver.add_constraint_on_nb_allocation_changes(
            ref=current_solution, nb_changes=self.nb_changes
        )


class NbUsagesAllocationConstraintExtractor(BaseConstraintExtractor[Task]):
    def __init__(
        self,
        plus_delta_nb_usages_total: int = 5,
        plus_delta_nb_usages_per_unary_resource: int = 3,
        minus_delta_nb_usages_per_unary_resource: int = 3,
    ):
        self.plus_delta_nb_usages_per_unary_resource = (
            plus_delta_nb_usages_per_unary_resource
        )
        self.minus_delta_nb_usages_per_unary_resource = (
            minus_delta_nb_usages_per_unary_resource
        )
        self.plus_delta_nb_usages_total = plus_delta_nb_usages_total

    def add_constraints(
        self,
        current_solution: TasksSolution[Task],
        solver: TasksCpSolver[Task],
        tasks_primary: set[Task],
        tasks_secondary: set[Task],
        params_constraint_builder: ParamsConstraintBuilder = None,
        **kwargs: Any,
    ) -> list[Any]:
        if not (
            isinstance(current_solution, AllocationSolution)
            and isinstance(solver, AllocationCpSolver)
        ):
            raise ValueError(
                "current_solution and solver must manage resource allocation."
            )
        constraints = []
        nb_usages_total = current_solution.compute_nb_unary_resource_usages()
        constraints += solver.add_constraint_on_total_nb_usages(
            SignEnum.LEQ, nb_usages_total + self.plus_delta_nb_usages_total
        )
        for unary_resource in current_solution.problem.unary_resources_list:
            nb_usages = current_solution.compute_nb_unary_resource_usages(
                unary_resources=(unary_resource,)
            )
            constraints += solver.add_constraint_on_unary_resource_nb_usages(
                unary_resource=unary_resource,
                sign=SignEnum.LEQ,
                target=nb_usages + self.plus_delta_nb_usages_per_unary_resource,
            )
            constraints += solver.add_constraint_on_unary_resource_nb_usages(
                unary_resource=unary_resource,
                sign=SignEnum.UEQ,
                target=nb_usages - self.minus_delta_nb_usages_per_unary_resource,
            )

        return constraints


def build_default_constraint_extractor(
    problem: TasksProblem, params_constraint_builder: ParamsConstraintBuilder
) -> BaseConstraintExtractor:
    extractors = []
    if isinstance(problem, SchedulingProblem):
        extractors.append(SchedulingConstraintExtractor())
    if isinstance(problem, MultimodeProblem) and problem.is_multimode:
        extractors.append(MultimodeConstraintExtractor())
    if isinstance(problem, AllocationProblem):
        if params_constraint_builder.first_method_multiskill:
            extractors.append(
                SubtasksAllocationConstraintExtractor(
                    fix_secondary_tasks_modes=params_constraint_builder.except_assigned_multiskill_primary_set,
                    frac_random_fixed_tasks=params_constraint_builder.fraction_of_task_assigned_multiskill,
                )
            )
            if params_constraint_builder.additional_methods:
                extractors += [
                    ChainingConstraintExtractor(),
                    NbUsagesAllocationConstraintExtractor(),
                ]
        elif params_constraint_builder.second_method_multiskill:
            extractors.append(SubresourcesAllocationConstraintExtractor())
            if params_constraint_builder.additional_methods:
                extractors += [
                    ChainingConstraintExtractor(),
                    NbChangesAllocationConstraintExtractor(nb_changes=10),
                ]
        else:
            extractors += [
                ChainingConstraintExtractor(),
                NbChangesAllocationConstraintExtractor(nb_changes=20),
            ]
    return ConstraintExtractorList(extractors=extractors)


class ObjectiveSubproblem(Enum):
    MAKESPAN_SUBTASKS = 0
    SUM_START_SUBTASKS = 1
    SUM_END_SUBTASKS = 2
    GLOBAL_MAKESPAN = 3


class BaseTasksConstraintHandler(ConstraintHandler, Generic[Task]):
    """Generic constraint handler for tasks related problem."""

    def __init__(
        self,
        problem: TasksProblem,
        neighbor_builder: Optional[NeighborBuilder[Task]] = None,
        constraints_extractor: Optional[BaseConstraintExtractor] = None,
        params_constraint_builder: Optional[ParamsConstraintBuilder] = None,
    ):
        self.problem = problem
        if neighbor_builder is None:
            self.neighbor_builder = build_default_neighbor_builder(problem=problem)
        else:
            self.neighbor_builder = neighbor_builder
        if params_constraint_builder is None:
            self.params_constraint_builder = ParamsConstraintBuilder.default()
        else:
            self.params_constraint_builder = params_constraint_builder
        if constraints_extractor is None:
            self.constraints_extractor = build_default_constraint_extractor(
                problem=problem,
                params_constraint_builder=self.params_constraint_builder,
            )
        else:
            self.constraints_extractor = constraints_extractor

    def adding_constraint_from_results_store(
        self,
        solver: SchedulingCpSolver,
        result_storage: ResultStorage,
        **kwargs: Any,
    ) -> Iterable[Any]:
        # current solution
        current_solution: TasksSolution
        current_solution = result_storage.get_best_solution()
        # split tasks
        (tasks_primary, tasks_secondary) = self.neighbor_builder.find_subtasks(
            current_solution=current_solution
        )
        logger.debug(self.__class__.__name__)
        logger.debug(
            f"{len(tasks_primary)} in first set, {len(tasks_secondary)} in second set"
        )

        constraints = self.constraints_extractor.add_constraints(
            solver=solver,
            current_solution=current_solution,
            tasks_primary=tasks_primary,
            tasks_secondary=tasks_secondary,
            params_constraint_builder=self.params_constraint_builder,
        )
        return constraints


class SchedulingConstraintHandler(BaseTasksConstraintHandler[Task]):
    """Generic constraint handler for scheduling problems.

    Include constraints for multimode and allocation features if present.

    """

    def __init__(
        self,
        problem: SchedulingProblem,
        neighbor_builder: Optional[NeighborBuilder[Task]] = None,
        constraints_extractor: Optional[BaseConstraintExtractor] = None,
        params_constraint_builder: Optional[ParamsConstraintBuilder] = None,
        objective_subproblem: ObjectiveSubproblem = ObjectiveSubproblem.GLOBAL_MAKESPAN,
    ):
        super().__init__(
            problem=problem,
            neighbor_builder=neighbor_builder,
            constraints_extractor=constraints_extractor,
            params_constraint_builder=params_constraint_builder,
        )
        self.objective_subproblem = objective_subproblem

    def adding_constraint_from_results_store(
        self,
        solver: TasksCpSolver,
        result_storage: ResultStorage,
        **kwargs: Any,
    ) -> Iterable[Any]:
        # current solution
        current_solution = result_storage.get_best_solution()

        if not (
            isinstance(current_solution, SchedulingSolution)
            and isinstance(solver, SchedulingCpSolver)
        ):
            raise ValueError(
                f"{self.__class__.__name__} extract constraints only "
                f"if solution and solver are related to a scheduling problem."
            )

        # split tasks
        (tasks_primary, tasks_secondary) = self.neighbor_builder.find_subtasks(
            current_solution=current_solution
        )
        logger.debug(self.__class__.__name__)
        logger.debug(
            f"{len(tasks_primary)} in first set, {len(tasks_secondary)} in second set"
        )

        constraints = self.constraints_extractor.add_constraints(
            solver=solver,
            current_solution=current_solution,
            tasks_primary=tasks_primary,
            tasks_secondary=tasks_secondary,
            params_constraint_builder=self.params_constraint_builder,
        )

        # change objective and add constraint on it
        if self.objective_subproblem == ObjectiveSubproblem.MAKESPAN_SUBTASKS:
            objective = solver.get_subtasks_makespan_variable(subtasks=tasks_primary)
            solver.minimize_variable(objective)
            current_max = max([current_solution.get_end_time(t) for t in tasks_primary])
            constraints += solver.add_bound_constraint(
                var=objective, sign=SignEnum.LEQ, value=current_max
            )
        elif self.objective_subproblem == ObjectiveSubproblem.GLOBAL_MAKESPAN:
            objective = solver.get_global_makespan_variable()
            solver.minimize_variable(objective)
        elif self.objective_subproblem == ObjectiveSubproblem.SUM_START_SUBTASKS:
            objective = solver.get_subtasks_sum_start_time_variable(
                subtasks=tasks_primary
            )
            solver.minimize_variable(objective)
            sum_start = sum(
                [
                    1  # (10 if t == self.problem.sink_task else 1)
                    * current_solution.get_start_time(t)
                    for t in tasks_primary
                ]
            )
            constraints += solver.add_bound_constraint(
                var=objective, sign=SignEnum.LEQ, value=sum_start
            )
        elif self.objective_subproblem == ObjectiveSubproblem.SUM_END_SUBTASKS:
            objective = solver.get_subtasks_sum_end_time_variable(
                subtasks=tasks_primary
            )
            solver.minimize_variable(objective)
            sum_start = sum(
                [
                    1  # (10 if t == self.problem.sink_task else 1)
                    * current_solution.get_end_time(t)
                    for t in tasks_primary
                ]
            )
            constraints += solver.add_bound_constraint(
                var=objective, sign=SignEnum.LEQ, value=sum_start
            )

        return constraints
