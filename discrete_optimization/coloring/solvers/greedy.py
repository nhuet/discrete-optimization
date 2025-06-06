"""Greedy solvers for coloring problem : binding from networkx library methods."""

#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from enum import Enum
from typing import Any, Optional

import networkx as nx

from discrete_optimization.coloring.problem import ColoringProblem, ColoringSolution
from discrete_optimization.coloring.solvers import ColoringSolver
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    EnumHyperparameter,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

logger = logging.getLogger(__name__)


strategies = [
    "largest_first",
    "random_sequential",
    "smallest_last",
    "independent_set",
    "connected_sequential_dfs",
    "connected_sequential_bfs",
    "connected_sequential",
    "saturation_largest_first",
    "DSATUR",
]


class NxGreedyColoringMethod(Enum):
    largest_first = "largest_first"
    random_sequential = "random_sequential"
    smallest_last = "smallest_last"
    independent_set = "independent_set"
    connected_sequential_dfs = "connected_sequential_dfs"
    connected_sequential_bfs = "connected_sequential_bfs"
    connected_sequential = "connected_sequential"
    saturation_largest_first = "saturation_largest_first"
    dsatur = "DSATUR"
    best = "best"


class GreedyColoringSolver(ColoringSolver):
    """Binded solver of networkx heuristics for coloring problem."""

    hyperparameters = [
        EnumHyperparameter(
            name="strategy",
            enum=NxGreedyColoringMethod,
            default=NxGreedyColoringMethod.best,
        )
    ]

    def __init__(
        self,
        problem: ColoringProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.nx_graph = self.problem.graph.to_networkx()

    def solve(self, **kwargs: Any) -> ResultStorage:
        """Run the greedy solver for the given problem.

        Keyword Args:
            strategy (NxGreedyColoringMethod) : one of the method used by networkx to compute coloring solution,
                                                or use NXGreedyColoringMethod.best to run each of them and return
                                                the best result.
            verbose (bool)


        Returns:
            results (ResultStorage) : storage of solution found by the greedy solver.

        """
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        greedy_strategy: NxGreedyColoringMethod = kwargs["strategy"]
        strategy_name = greedy_strategy.value
        if strategy_name == "best":
            strategies_to_test = strategies
        else:
            strategies_to_test = [strategy_name]
        best_solution = None
        best_nb_color = float("inf")
        for strategy in strategies_to_test:
            try:
                colors = nx.algorithms.coloring.greedy_color(
                    self.nx_graph, strategy=strategy, interchange=False
                )
                # number_colors = len(set(list(colors.values())))
                raw_solution = [colors[i] for i in self.problem.nodes_name]
                number_colors = self.problem.count_colors(raw_solution)
                logger.info(f"{strategy} : number colors : {number_colors}")
                if number_colors < best_nb_color:
                    best_solution = raw_solution
                    best_nb_color = number_colors
            except Exception as e:
                logger.info(f"Failed strategy : {strategy} {e}")
        logger.info(f"best found : {best_nb_color}")
        solution = ColoringSolution(self.problem, colors=best_solution, nb_color=None)
        solution = solution.to_reformated_solution()
        fit = self.aggreg_from_sol(solution)
        logger.debug(f"Solution found : {solution, fit}")
        return self.create_result_storage([(solution, fit)])
