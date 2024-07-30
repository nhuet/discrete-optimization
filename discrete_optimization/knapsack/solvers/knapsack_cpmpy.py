#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
# Thanks to Leuven university for the cpmyp library.
from typing import Any, Dict, Optional

from cpmpy import Model, boolvar

from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.knapsack.knapsack_model import (
    KnapsackModel,
    KnapsackSolution,
)
from discrete_optimization.knapsack.solvers.knapsack_solver import SolverKnapsack


class CPMPYKnapsackSolver(SolverKnapsack):
    def __init__(
        self,
        problem: KnapsackModel,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.model: Optional[Model] = None
        self.variables: Dict[str, Any] = {}

    def init_model(self, **kwargs: Any) -> None:
        values = [
            self.problem.list_items[i].value for i in range(self.problem.nb_items)
        ]
        weights = [
            self.problem.list_items[i].weight for i in range(self.problem.nb_items)
        ]
        capacity = self.problem.max_capacity
        # Construct the model.
        x = boolvar(shape=self.problem.nb_items, name="x")
        self.model = Model(sum(x * weights) <= capacity, maximize=sum(x * values))
        self.variables["x"] = x

    def solve(
        self, parameters_cp: Optional[ParametersCP] = None, **kwargs: Any
    ) -> ResultStorage:
        if parameters_cp is None:
            parameters_cp = ParametersCP.default()
        if self.model is None:
            self.init_model()
            if self.model is None:  # for mypy
                raise RuntimeError(
                    "self.model must not be None after self.init_model()."
                )
        self.model.solve(
            kwargs.get("solver", "ortools"), time_limit=parameters_cp.time_limit
        )
        list_taken = self.variables["x"].value()
        sol = KnapsackSolution(problem=self.problem, list_taken=list_taken)
        fit = self.aggreg_from_sol(sol)
        return self.create_result_storage(
            [(sol, fit)],
        )
