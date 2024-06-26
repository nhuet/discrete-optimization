#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import os
from enum import Enum
from typing import Any, List, Optional, Tuple, Union

from deprecation import deprecated
from minizinc import Instance, Model, Result, Solver

from discrete_optimization.facility.facility_model import (
    FacilityProblem,
    FacilitySolution,
)
from discrete_optimization.facility.solvers.facility_lp_solver import (
    compute_length_matrix,
)
from discrete_optimization.facility.solvers.facility_solver import SolverFacility
from discrete_optimization.facility.solvers.greedy_solvers import (
    GreedySolverDistanceBased,
)
from discrete_optimization.generic_tools.cp_tools import (
    CPSolverName,
    MinizincCPSolver,
    ParametersCP,
    find_right_minizinc_solver_name,
    map_cp_solver_name,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
    TupleFitness,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    EnumHyperparameter,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

logger = logging.getLogger(__name__)


path_minizinc = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../minizinc/")
)


class FacilityCPModel(Enum):
    DEFAULT_INT = 0
    DEFAULT_INT_LNS = 1


file_dict = {
    FacilityCPModel.DEFAULT_INT: "facility_int.mzn",
    FacilityCPModel.DEFAULT_INT_LNS: "facility_int_lns.mzn",
}


class FacilitySolCP:
    objective: int
    __output_item: Optional[str] = None

    def __init__(self, objective: int, _output_item: Optional[str], **kwargs: Any):
        self.objective = objective
        self.dict = kwargs
        logger.debug(f"One solution {self.objective}")
        logger.debug(f"Output {_output_item}")

    def check(self) -> bool:
        return True


class FacilityCP(MinizincCPSolver, SolverFacility):
    """CP solver linked with minizinc implementation of coloring problem.

    Attributes:
        problem (FacilityProblem): facility problem instance to solve
        params_objective_function (ParamsObjectiveFunction): params of the objective function
        cp_solver_name (CPSolverName): backend solver to use with minizinc
        silent_solve_error (bool): if True, raise a warning instead of an error if the underlying instance.solve() crashes
        **args: unused
    """

    hyperparameters = [
        EnumHyperparameter(
            "cp_model", enum=FacilityCPModel, default=FacilityCPModel.DEFAULT_INT
        ),
        EnumHyperparameter(
            "cp_solver_name", enum=CPSolverName, default=CPSolverName.CHUFFED
        ),
    ]

    def __init__(
        self,
        problem: FacilityProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        silent_solve_error: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.silent_solve_error = silent_solve_error
        self.custom_output_type = False
        self.model: Optional[Model] = None

    def init_model(self, **kwargs: Any) -> None:
        """Initialise the minizinc instance to solve for a given instance.

        Keyword Args:
            cp_model (FacilityCPModel): CP model version
            object_output (bool): specify if the solution are returned in a FacilitySolCP object
                                  or native minizinc output.

        Returns: None

        """
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        model_type = kwargs["cp_model"]
        cp_solver_name = kwargs["cp_solver_name"]
        object_output = kwargs.get("object_output", True)
        path = os.path.join(path_minizinc, file_dict[model_type])
        self.model = Model(path)
        if object_output:
            self.model.output_type = FacilitySolCP
            self.custom_output_type = True
        solver = Solver.lookup(find_right_minizinc_solver_name(cp_solver_name))
        instance = Instance(solver, self.model)
        instance["nb_facilities"] = self.problem.facility_count
        instance["nb_customers"] = self.problem.customer_count
        setup_costs, closests, distances = compute_length_matrix(self.problem)
        if model_type in [FacilityCPModel.DEFAULT_INT, FacilityCPModel.DEFAULT_INT_LNS]:
            distances_list = [
                [int(distances[f, c]) for c in range(distances.shape[1])]
                for f in range(distances.shape[0])
            ]
            instance["distance"] = distances_list
            instance["setup_cost_vector"] = [int(s) for s in setup_costs]
            instance["demand"] = [
                int(self.problem.customers[c].demand)
                for c in range(self.problem.customer_count)
            ]
            instance["capacity"] = [
                int(self.problem.facilities[f].capacity)
                for f in range(self.problem.facility_count)
            ]
        else:
            distances_list = [
                [distances[f, c] for c in range(distances.shape[1])]
                for f in range(distances.shape[0])
            ]
            instance["distance"] = distances_list
            instance["setup_cost_vector"] = [s for s in setup_costs]
            instance["demand"] = [
                self.problem.customers[c].demand
                for c in range(self.problem.customer_count)
            ]
            instance["capacity"] = [
                self.problem.facilities[f].capacity
                for f in range(self.problem.facility_count)
            ]
        self.instance = instance

    def retrieve_solutions(
        self, result: Result, parameters_cp: ParametersCP
    ) -> ResultStorage:
        intermediate_solutions = parameters_cp.intermediate_solution
        list_facility = []
        objectives = []
        if intermediate_solutions:
            for i in range(len(result)):
                if not self.custom_output_type:
                    list_facility.append(result[i, "facility_for_customer"])
                    objectives.append(result[i, "objective"])
                else:
                    list_facility.append(result[i].dict["facility_for_customer"])
                    objectives.append(result[i].objective)
        else:
            if not self.custom_output_type:
                list_facility.append(result["facility_for_customer"])
                objectives.append(result["objective"])
            else:
                list_facility.append(result.dict["facility_for_customer"])
                objectives.append(result.objective)
        list_solutions_fit: List[Tuple[Solution, Union[float, TupleFitness]]] = []
        for facility, objective in zip(list_facility, objectives):
            facility_sol = FacilitySolution(self.problem, [f - 1 for f in facility])
            fit = self.aggreg_from_sol(facility_sol)
            list_solutions_fit.append((facility_sol, fit))
        return ResultStorage(
            list_solution_fits=list_solutions_fit,
            best_solution=None,
            mode_optim=self.params_objective_function.sense_function,
        )

    @deprecated(
        deprecated_in="0.1", details="Use rather initial solution provider utilities"
    )
    def get_solution(self, **kwargs: Any) -> FacilitySolution:
        greedy_start = kwargs.get("greedy_start", True)
        if greedy_start:
            logger.info("Computing greedy solution")
            greedy_solver = GreedySolverDistanceBased(self.problem)
            result = greedy_solver.solve()
            solution = result.get_best_solution()
            if solution is None:
                raise RuntimeError(
                    "greedy_solver.solve().get_best_solution() " "should not be None."
                )
            if not isinstance(solution, FacilitySolution):
                raise RuntimeError(
                    "greedy_solver.solve().get_best_solution() "
                    "should be a FacilitySolution."
                )
        else:
            logger.info("Get dummy solution")
            solution = self.problem.get_dummy_solution()
        logger.info("Greedy Done")
        return solution
