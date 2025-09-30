from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from typing import Any

from discrete_optimization.generic_scheduling_tools.base import (
    Task,
    TasksCpSolver,
    TasksProblem,
    TasksSolution,
)
from discrete_optimization.generic_scheduling_tools.enums import StartOrEnd
from discrete_optimization.generic_tools.cp_tools import SignEnum


class SchedulingProblem(TasksProblem[Task]):
    """Base class for scheduling problems.

    A scheduling problems is about finding start and end times to tasks.

    """

    def get_last_tasks(self) -> list[Task]:
        """Get a sublist of tasks that are candidate to be the last one scheduled.

        Default to all tasks.

        """
        return self.tasks_list


class SchedulingSolution(TasksSolution[Task]):
    """Base class for solution to scheduling problems."""

    problem: SchedulingProblem[Task]

    @abstractmethod
    def get_end_time(self, task: Task) -> int:
        ...

    @abstractmethod
    def get_start_time(self, task: Task) -> int:
        ...

    def get_max_end_time(self) -> int:
        return max(self.get_end_time(task) for task in self.problem.get_last_tasks())

    def constraint_on_task_satisfied(
        self, task: Task, start_or_end: StartOrEnd, sign: SignEnum, time: int
    ) -> bool:
        if start_or_end == StartOrEnd.START:
            actual_time = self.get_start_time(task)
        else:
            actual_time = self.get_end_time(task)

        if sign == SignEnum.UEQ:
            return actual_time >= time
        elif sign == SignEnum.LEQ:
            return actual_time <= time
        elif sign == SignEnum.LESS:
            return actual_time < time
        elif sign == SignEnum.UP:
            return actual_time > time
        elif sign == SignEnum.EQUAL:
            return actual_time == time

    def constraint_chaining_tasks_satisfied(self, task1: Task, task2: Task) -> bool:
        return self.get_end_time(task1) == self.get_start_time(task2)


class SchedulingCpSolver(TasksCpSolver[Task]):
    """Base class for cp solvers handling scheduling problems."""

    problem: SchedulingProblem[Task]

    @abstractmethod
    def add_constraint_on_task(
        self, task: Task, start_or_end: StartOrEnd, sign: SignEnum, time: int
    ) -> list[Any]:
        """Add constraint on given task start or end

        task start or end must compare to `time` according to `sign`

        Args:
            task:
            start_or_end:
            sign:
            time:

        Returns:
            resulting constraints
        """
        ...

    @abstractmethod
    def add_constraint_chaining_tasks(self, task1: Task, task2: Task) -> list[Any]:
        """Add constraint chaining task1 with task2

        task2 start == task1 end

        Args:
            task1:
            task2:

        Returns:
            resulting constraints

        """
        ...

    def add_constraint_on_max_end_time(self, sign: SignEnum, time: int) -> list[Any]:
        """Add constraint on max end time of all tasks.

        Default implementation add constraints on all end time, and works only for sign < or <=.
        But depending on problems, it can be simplified in inherited classes (like using sink task in rcpsp).

        """
        if sign not in (SignEnum.LEQ, SignEnum.LESS):
            raise NotImplementedError(
                "Default implementation only available for < or <= constraint on max_end_time."
            )
        constraints = []
        for task in self.problem.get_last_tasks():
            constraints += self.add_constraint_on_task(
                task=task, start_or_end=StartOrEnd.END, sign=sign, time=time
            )
        return constraints

    @abstractmethod
    def add_bound_constraint(self, var: Any, sign: SignEnum, value: int) -> list[Any]:
        """Add constraint of bound type on an integer variable (or expression) of the underlying cp model.

        `var` must compare to `value` according to `value`.

        Args:
            var:
            sign:
            value:

        Returns:

        """
        ...

    def set_objective_max_end_time(self) -> Any:
        """Set the internal objective of the cp solver to be the global makespan.

        Default implementation uses `set_objective_max_end_time_substasks` on all tasks.

        Args:
            subtasks:

        Returns:
            objective variable to minimize

        """
        return self.set_objective_max_end_time_substasks(
            subtasks=set(self.problem.get_last_tasks())
        )

    @abstractmethod
    def set_objective_max_end_time_substasks(self, subtasks: Iterable[Task]) -> Any:
        """Set the internal objective of the cp solver to be the makespan on a subset of tasks.

        Args:
            subtasks:

        Returns:
            objective variable to minimize

        """
        ...

    @abstractmethod
    def set_objective_sum_end_time_substasks(self, subtasks: Iterable[Task]) -> Any:
        """Set the internal objective of the cp solver to be sum of end times on a subset of tasks.

        Args:
            subtasks:

        Returns:
            objective variable to minimize

        """
        ...

    @abstractmethod
    def set_objective_sum_start_time_substasks(self, subtasks: Iterable[Task]) -> Any:
        """Set the internal objective of the cp solver to be the sum of start times on a subset of tasks.

        Args:
            subtasks:

        Returns:
            objective variable to minimize

        """
        ...
