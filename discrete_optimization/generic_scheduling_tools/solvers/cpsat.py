from abc import abstractmethod
from typing import Any, Iterable, Optional

from ortools.sat.python.cp_model import IntVar, LinearExprT

from discrete_optimization.generic_scheduling_tools.allocation import (
    AllocationCpSolver,
    AllocationSolution,
    UnaryResource,
)
from discrete_optimization.generic_scheduling_tools.enums import StartOrEnd
from discrete_optimization.generic_scheduling_tools.multimode import MultimodeCpSolver
from discrete_optimization.generic_scheduling_tools.scheduling import (
    SchedulingCpSolver,
    Task,
)
from discrete_optimization.generic_tools.cp_tools import SignEnum
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver


class SchedulingCpSatSolver(OrtoolsCpSatSolver, SchedulingCpSolver[Task]):
    """Base class for most ortools/cpsat solvers handling scheduling problems.

    Allows to have common code.

    """

    _makespan: Optional[IntVar] = None
    """Internal variable use to define the objective (global or partial makespan)."""

    constraints_on_makespan: Optional[list[Any]] = None
    """Constraints on makespan so that it can be considered as the objective."""

    @abstractmethod
    def get_task_start_or_end_variable(
        self, task: Task, start_or_end: StartOrEnd
    ) -> LinearExprT:
        """Retrieve the variable storing the start or end time of given task.

        Args:
            task:
            start_or_end:

        Returns:

        """
        ...

    def add_constraint_on_task(
        self, task: Task, start_or_end: StartOrEnd, sign: SignEnum, time: int
    ) -> list[Any]:
        var = self.get_task_start_or_end_variable(task=task, start_or_end=start_or_end)
        return self.add_bound_constraint(var=var, sign=sign, value=time)

    def add_constraint_chaining_tasks(self, task1: Task, task2: Task) -> list[Any]:
        var1 = self.get_task_start_or_end_variable(
            task=task1, start_or_end=StartOrEnd.END
        )
        var2 = self.get_task_start_or_end_variable(
            task=task2, start_or_end=StartOrEnd.START
        )
        return [self.cp_model.add(var1 == var2)]

    def get_makespan_var(self) -> IntVar:
        """Get the makespan variable used to track global or subtasks makespan."""
        if self._makespan is None:
            self._makespan = self.cp_model.NewIntVar(
                lb=self.get_makespan_lower_bound(),
                ub=self.get_makespan_upper_bound(),
                name="makespan",
            )
        return self._makespan

    def get_makespan_lower_bound(self) -> int:
        """Get a lower bound on global makespan.

        Can be overriden in solvers wanting to specify it in init_model() for instance.

        """
        return self.problem.get_makespan_lower_bound()

    def get_makespan_upper_bound(self) -> int:
        """Get a upper bound on global makespan."""
        return self.problem.get_makespan_upper_bound()

    def remove_constraints_on_objective(self) -> None:
        if self.constraints_on_makespan is not None:
            self.remove_constraints(self.constraints_on_makespan)

    def get_subtasks_makespan_variable(self, subtasks: Iterable[Task]) -> Any:
        # remove previous constraints on makespan variable from cp model
        self.remove_constraints_on_objective()
        # get makespan variable
        makespan = self.get_makespan_var()
        # update those constraints
        self.constraints_on_makespan = [
            self.cp_model.AddMaxEquality(
                makespan,
                [
                    self.get_task_start_or_end_variable(task, StartOrEnd.END)
                    for task in subtasks
                ],
            )
        ]
        return makespan

    def get_subtasks_sum_end_time_variable(self, subtasks: Iterable[Task]) -> Any:
        self.remove_constraints_on_objective()
        return sum(
            self.get_task_start_or_end_variable(task, StartOrEnd.END)
            for task in subtasks
        )

    def get_subtasks_sum_start_time_variable(self, subtasks: Iterable[Task]) -> Any:
        self.remove_constraints_on_objective()
        return sum(
            self.get_task_start_or_end_variable(task, StartOrEnd.START)
            for task in subtasks
        )

    def minimize_variable(self, var: Any) -> None:
        self.cp_model.minimize(var)


class MultimodeCpSatSolver(OrtoolsCpSatSolver, MultimodeCpSolver[Task]):
    @abstractmethod
    def get_task_mode_is_present_variable(self, task: Task, mode: int) -> LinearExprT:
        """Retrieve the 0-1 variable/expression telling if the mode is used for the task.

        Args:
            task:
            mode:

        Returns:

        """
        ...

    def add_constraint_on_task_mode(self, task: Task, mode: int) -> list[Any]:
        possible_modes = self.problem.get_task_modes(task)
        if mode not in possible_modes:
            raise ValueError(f"Task {task} cannot be done with mode {mode}.")
        if len(possible_modes) == 1:
            return []
        constraints = []
        for other_mode in possible_modes:
            var = self.get_task_mode_is_present_variable(task=task, mode=other_mode)
            if other_mode == mode:
                constraints.append(self.cp_model.add(var == True))
            else:
                constraints.append(self.cp_model.add(var == False))
        return constraints


class AllocationCpSatSolver(
    OrtoolsCpSatSolver,
    AllocationCpSolver[Task, UnaryResource],
):

    allocation_changes_variables_created = False
    allocation_changes_variables: dict[tuple[Task, UnaryResource], IntVar]

    @abstractmethod
    def get_task_unary_resource_is_present_variable(
        self, task: Task, unary_resource: UnaryResource
    ) -> LinearExprT:
        """Return a 0-1 variable/expression telling if the unary_resource is used for the task.

        NB: sometimes the given resource is never to be used by a task and the variable has not been created.
        The convention is to return 0 in that case.

        """
        ...

    def add_constraint_on_task_unary_resource_allocation(
        self, task: Task, unary_resource: UnaryResource, used: bool
    ) -> list[Any]:
        var = self.get_task_unary_resource_is_present_variable(
            task=task, unary_resource=unary_resource
        )
        return [self.cp_model.add(var == used)]

    def add_constraint_on_nb_allocation_changes(
        self, ref: AllocationSolution[Task, UnaryResource], nb_changes: int
    ) -> list[Any]:
        tasks, unary_resources = self.get_default_tasks_n_unary_resources()
        self.create_allocation_changes_variables()
        # constraints so that change variables reflect diff to ref
        constraints = [
            self.cp_model.add(
                self.get_task_unary_resource_is_present_variable(
                    task=task, unary_resource=unary_resource
                )
                != ref.is_allocated(task=task, unary_resource=unary_resource)
            ).only_enforce_if(self.allocation_changes_variables[(task, unary_resource)])
            for task in tasks
            for unary_resource in unary_resources
        ] + [
            self.cp_model.add(
                self.get_task_unary_resource_is_present_variable(
                    task=task, unary_resource=unary_resource
                )
                == ref.is_allocated(task=task, unary_resource=unary_resource)
            ).only_enforce_if(
                ~self.allocation_changes_variables[(task, unary_resource)]
            )
            for task in tasks
            for unary_resource in unary_resources
        ]
        # nb of changes variable
        var = sum(
            self.allocation_changes_variables[(task, unary_resource)]
            for task in tasks
            for unary_resource in unary_resources
        )
        return [self.cp_model.add(var <= nb_changes)]

    def create_allocation_changes_variables(self):
        """Create variables necessary for constraint on nb of changes."""
        if not self.allocation_changes_variables_created:
            tasks, unary_resources = self.get_default_tasks_n_unary_resources()
            self.allocation_changes_variables = {
                (task, unary_resource): self.cp_model.new_bool_var(
                    f"change_{task}_{unary_resource}"
                )
                for task in tasks
                for unary_resource in unary_resources
            }
            self.allocation_changes_variables_created = True

    def add_constraint_nb_unary_resource_usages(
        self,
        sign: SignEnum,
        target: int,
        tasks: Optional[Iterable[Task]] = None,
        unary_resources: Optional[Iterable[UnaryResource]] = None,
    ) -> list[Any]:
        tasks, unary_resources = self.get_default_tasks_n_unary_resources(
            tasks=tasks, unary_resources=unary_resources
        )
        var = sum(
            self.get_task_unary_resource_is_present_variable(task, unary_resource)
            for task in tasks
            for unary_resource in unary_resources
        )
        return self.add_bound_constraint(var=var, sign=sign, value=target)

    def add_constraint_on_total_nb_usages(
        self, sign: SignEnum, target: int
    ) -> list[Any]:
        return self.add_constraint_nb_unary_resource_usages(sign=sign, target=target)

    def add_constraint_on_unary_resource_nb_usages(
        self, unary_resource: UnaryResource, sign: SignEnum, target: int
    ) -> list[Any]:
        return self.add_constraint_nb_unary_resource_usages(
            sign=sign, target=target, unary_resources=(unary_resource,)
        )
