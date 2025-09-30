from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from typing import Any, Generic, Optional, TypeVar

from discrete_optimization.generic_scheduling_tools.base import (
    Task,
    TasksCpSolver,
    TasksProblem,
    TasksSolution,
)
from discrete_optimization.generic_tools.cp_tools import SignEnum

UnaryResource = TypeVar("UnaryResource")


class AllocationSolution(TasksSolution[Task], Generic[Task, UnaryResource]):
    """Class inherited by a solution for allocation problems."""

    problem: AllocationProblem[Task, UnaryResource]

    @abstractmethod
    def is_allocated(self, task: Task, unary_resource: UnaryResource) -> bool:
        """Return the usage of the unary resource for the given task.

        Args:
            task:
            unary_resource:

        Returns:

        """
        ...

    def get_default_tasks_n_unary_resources(
        self,
        tasks: Optional[Iterable[Task]] = None,
        unary_resources: Optional[Iterable[UnaryResource]] = None,
    ) -> tuple[Iterable[Task], Iterable[UnaryResource]]:
        return get_default_tasks_n_unary_resources(self.problem, tasks, unary_resources)

    def compute_nb_unary_resource_usages(
        self,
        tasks: Optional[Iterable[Task]] = None,
        unary_resources: Optional[Iterable[UnaryResource]] = None,
    ):
        tasks, unary_resources = self.get_default_tasks_n_unary_resources(
            tasks, unary_resources
        )
        return sum(
            self.is_allocated(task=task, unary_resource=unary_resource)
            for task in tasks
            for unary_resource in unary_resources
        )

    def _check_same_problem(self, ref: AllocationSolution[Task, UnaryResource]) -> None:
        if self.problem is not ref.problem:
            raise ValueError("We can compare only solutions for same problem.")

    def compute_nb_allocation_changes(
        self, ref: AllocationSolution[Task, UnaryResource]
    ) -> int:
        self._check_same_problem(ref)
        return sum(
            self.is_allocated(task=task, unary_resource=unary_resource)
            != ref.is_allocated(task=task, unary_resource=unary_resource)
            for task in self.problem.tasks_list
            for unary_resource in self.problem.unary_resources_list
        )

    def check_same_allocation_as_ref(
        self,
        ref: AllocationSolution[Task, UnaryResource],
        tasks: Optional[Iterable[Task]] = None,
        unary_resources: Optional[Iterable[UnaryResource]] = None,
    ) -> bool:
        self._check_same_problem(ref)
        tasks, unary_resources = self.get_default_tasks_n_unary_resources(
            tasks, unary_resources
        )
        return all(
            self.is_allocated(task=task, unary_resource=unary_resource)
            == ref.is_allocated(task=task, unary_resource=unary_resource)
            for task in tasks
            for unary_resource in unary_resources
        )


class AllocationProblem(TasksProblem[Task], Generic[Task, UnaryResource]):
    """Base class for allocation problems.

    An allocation problems consist in allocating resources to tasks.

    """

    unary_resources_list: list[UnaryResource]
    """Available unary resources.

    It can correspond to employees (rcpsp-multiskill), teams (workforce-scheduling), or
    a mix of several types.

    """


class AllocationCpSolver(TasksCpSolver[Task], Generic[Task, UnaryResource]):
    """Base class for solver managing constraints on allocation."""

    problem: AllocationProblem[Task, UnaryResource]

    @abstractmethod
    def add_constraint_on_task_unary_resource_allocation(
        self, task: Task, unary_resource: UnaryResource, used: bool
    ) -> list[Any]:
        """Add constraint on allocation of given unary resource for the given task

        Args:
            task:
            unary_resource:
            used: if True, we enforce the allocation of `unary_resource` to `task`, else we prevent it

        Returns:
            resulting constraints

        """
        ...

    @abstractmethod
    def add_constraint_on_nb_allocation_changes(
        self, ref: AllocationSolution[Task, UnaryResource], nb_changes: int
    ) -> list[Any]:
        """Add contraint on maximal number of allocation changes from the given reference.

        Args:
            ref:
            nb_changes: maximal number of changes

        Returns:
            resulting constraints

        """
        ...

    @abstractmethod
    def add_constraint_on_total_nb_usages(
        self, sign: SignEnum, target: int
    ) -> list[Any]:
        ...

    @abstractmethod
    def add_constraint_on_unary_resource_nb_usages(
        self, unary_resource: UnaryResource, sign: SignEnum, target: int
    ) -> list[Any]:
        ...

    def get_default_tasks_n_unary_resources(
        self,
        tasks: Optional[Iterable[Task]] = None,
        unary_resources: Optional[Iterable[UnaryResource]] = None,
    ) -> tuple[Iterable[Task], Iterable[UnaryResource]]:
        return get_default_tasks_n_unary_resources(self.problem, tasks, unary_resources)

    def add_constraint_same_allocation_as_ref(
        self,
        ref: AllocationSolution[Task, UnaryResource],
        tasks: Optional[Iterable[Task]] = None,
        unary_resources: Optional[Iterable[UnaryResource]] = None,
    ) -> list[Any]:
        """Add constraint to keep same allocation as the reference for the given tasks and unary resources subsets.

        Args:
            ref:
            tasks:
            unary_resources:

        Returns:
            resulting constraints

        """
        constraints = []
        tasks, unary_resources = self.get_default_tasks_n_unary_resources(
            tasks, unary_resources
        )
        for unary_resource in unary_resources:
            for task in tasks:
                constraints += self.add_constraint_on_task_unary_resource_allocation(
                    task=task,
                    unary_resource=unary_resource,
                    used=ref.is_allocated(task=task, unary_resource=unary_resource),
                )
        return constraints


def get_default_tasks_n_unary_resources(
    problem: AllocationProblem,
    tasks: Optional[Iterable[Task]] = None,
    unary_resources: Optional[Iterable[UnaryResource]] = None,
) -> tuple[Iterable[Task], Iterable[UnaryResource]]:
    if tasks is None:
        tasks = problem.tasks_list
    if unary_resources is None:
        unary_resources = problem.unary_resources_list
    return tasks, unary_resources
