#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from typing import Any, Dict, List, Optional

from ortools.sat.python.cp_model import CpModel, CpSolverSolutionCallback, IntVar

from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCPSatSolver
from discrete_optimization.knapsack.knapsack_model import (
    KnapsackModel,
    KnapsackSolution,
)
from discrete_optimization.knapsack.solvers.knapsack_solver import SolverKnapsack

logger = logging.getLogger(__name__)


class CPSatKnapsackSolver(OrtoolsCPSatSolver, SolverKnapsack):
    def __init__(
        self,
        problem: KnapsackModel,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.variables: Dict[str, List[IntVar]] = {}

    def init_model(self, **args: Any) -> None:
        """Init CP model."""
        model = CpModel()
        variables = [
            model.NewBoolVar(name=f"x_{i}") for i in range(self.problem.nb_items)
        ]
        self.variables["taken"] = variables
        self.cp_model = model

        model.Add(self._intern_weight() <= self.problem.max_capacity)

        self.set_model_objective("value")

    def _intern_value(self):
        variables = self.variables["taken"]
        return sum(
            [
                variables[i] * self.problem.list_items[i].value
                for i in range(self.problem.nb_items)
            ]
        )

    def _intern_weight(self):
        variables = self.variables["taken"]
        return sum(
            [
                variables[i] * self.problem.list_items[i].weight
                for i in range(self.problem.nb_items)
            ]
        )

    def _intern_heaviest_item(self):
        variables = self.variables["taken"]
        return max(
            [
                variables[i] * self.problem.list_items[i].weight
                for i in range(self.problem.nb_items)
            ]
        )

    def _intern_weight_violation(self):
        return max(0.0, self._intern_weight() - self.problem.max_capacity)

    def _intern_objective(self, obj: str) -> Any:
        intern_objective_mapping = {
            "value": self._intern_value,
            "weight": self._intern_weight,
            "heaviest_item": self._intern_heaviest_item,
            "weight_violation": self._intern_weight_violation,
        }
        if obj in intern_objective_mapping:
            intern_objective_mapping[obj]()
        else:
            raise ValueError(f"Unknown objective '{obj}'.")

    def set_model_objective(self, obj: str) -> None:
        self.cp_model.Maximize(self._intern_objective(obj))

    def add_model_constraint(self, obj: str, value: float) -> Any:
        return self.cp_model.Add(self._intern_objective(obj) >= value)

    def retrieve_solution(
        self, cpsolvercb: CpSolverSolutionCallback
    ) -> KnapsackSolution:
        """Construct a do solution from the cpsat solver internal solution.

        It will be called each time the cpsat solver find a new solution.
        At that point, value of internal variables are accessible via `cpsolvercb.Value(VARIABLE_NAME)`.

        Args:
            cpsolvercb: the ortools callback called when the cpsat solver finds a new solution.

        Returns:
            the intermediate solution, at do format.

        """
        taken = [int(cpsolvercb.Value(var)) for var in self.variables["taken"]]
        return KnapsackSolution(problem=self.problem, list_taken=taken)
