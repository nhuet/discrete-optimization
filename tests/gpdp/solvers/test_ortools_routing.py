#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import random
from datetime import datetime

import numpy as np
import pytest

from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
from discrete_optimization.generic_tools.callbacks.loggers import NbIterationTracker
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveHandling,
    ParamsObjectiveFunction,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.gpdp.builders.instance_builders import (
    GpdpProblem,
    create_selective_tsp,
)
from discrete_optimization.gpdp.problem import GpdpSolution
from discrete_optimization.gpdp.solvers.ortools_routing import (
    FirstSolutionStrategy,
    LocalSearchMetaheuristic,
    OrtoolsGpdpSolver,
    ParametersCost,
)

SEED = 42


@pytest.fixture()
def random_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    return SEED


nb_nodes = 1000
nb_vehicles = 1
nb_clusters = 100


def test_ortools_with_cb(random_seed):
    gpdp: GpdpProblem = create_selective_tsp(
        nb_nodes=nb_nodes, nb_vehicles=nb_vehicles, nb_clusters=nb_clusters
    )
    params_objective_function = ParamsObjectiveFunction(
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=["distance_max"],
        weights=[1],
        sense_function=ModeOptim.MINIMIZATION,
    )
    #  hyperparameters
    first_solution_strategy_name = "LOCAL_CHEAPEST_INSERTION"
    first_solution_strategy = FirstSolutionStrategy[first_solution_strategy_name]
    local_search_metaheuristic_name = "SIMULATED_ANNEALING"
    local_search_metaheuristic = LocalSearchMetaheuristic[
        local_search_metaheuristic_name
    ]
    first_solution_strategy = FirstSolutionStrategy[first_solution_strategy_name]
    use_lns = False
    use_cp = True
    use_cp_sat = True

    #  solver init
    solver = OrtoolsGpdpSolver(
        problem=gpdp,
        factor_multiplier_distance=1,
        factor_multiplier_time=1,
        params_objective_function=params_objective_function,
    )
    solver.init_model(
        one_visit_per_cluster=True,
        one_visit_per_node=False,
        include_time_dimension=True,
        include_demand=True,
        include_mandatory=True,
        include_pickup_and_delivery=False,
        parameters_cost=[ParametersCost(dimension_name="Distance", global_span=True)],
        local_search_metaheuristic=LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        first_solution_strategy=first_solution_strategy,
        time_limit=10,
        use_cp=use_cp,
        use_lns=use_lns,
        use_cp_sat=use_cp_sat,
    )

    # callbacks
    nb_iteration_tracker = NbIterationTracker()
    timer = TimerStopper(total_seconds=2, check_nb_steps=1)

    start_time = datetime.utcnow()
    #  solve
    sol, fit = solver.solve(
        callbacks=[nb_iteration_tracker, timer]
    ).get_best_solution_fit()
    end_time = datetime.utcnow()

    assert isinstance(fit, float)
    assert isinstance(sol, GpdpSolution)
    assert nb_iteration_tracker.nb_iteration > 0
    assert (end_time - start_time).total_seconds() < 10


def test_ortools_with_warm_start(random_seed):
    gpdp: GpdpProblem = create_selective_tsp(
        nb_nodes=nb_nodes, nb_vehicles=nb_vehicles, nb_clusters=nb_clusters
    )
    params_objective_function = ParamsObjectiveFunction(
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=["distance_max"],
        weights=[1],
        sense_function=ModeOptim.MINIMIZATION,
    )
    #  hyperparameters
    first_solution_strategy_name = "AUTOMATIC"
    first_solution_strategy = FirstSolutionStrategy[first_solution_strategy_name]
    use_lns = False
    use_cp = True
    use_cp_sat = True

    #  solver init
    solver = OrtoolsGpdpSolver(
        problem=gpdp,
        factor_multiplier_distance=1,
        factor_multiplier_time=1,
        params_objective_function=params_objective_function,
    )
    solver.init_model(
        one_visit_per_cluster=True,
        one_visit_per_node=False,
        include_time_dimension=True,
        include_demand=True,
        include_mandatory=True,
        include_pickup_and_delivery=False,
        parameters_cost=[ParametersCost(dimension_name="Distance", global_span=True)],
        local_search_metaheuristic=LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        first_solution_strategy=first_solution_strategy,
        time_limit=10,
        use_cp=use_cp,
        use_lns=use_lns,
        use_cp_sat=use_cp_sat,
    )

    # solve
    result_storage = solver.solve()
    print(len(result_storage))
    print([(fit, sol.trajectories) for sol, fit in result_storage])

    # warm start
    start_solution = result_storage[1][0]
    assert result_storage[0][0].trajectories != start_solution.trajectories

    solver = OrtoolsGpdpSolver(
        problem=gpdp,
        factor_multiplier_distance=1,
        factor_multiplier_time=1,
        params_objective_function=params_objective_function,
    )
    solver.init_model(
        one_visit_per_cluster=True,
        one_visit_per_node=False,
        include_time_dimension=True,
        include_demand=True,
        include_mandatory=True,
        include_pickup_and_delivery=False,
        parameters_cost=[ParametersCost(dimension_name="Distance", global_span=True)],
        local_search_metaheuristic=LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        first_solution_strategy=first_solution_strategy,
        time_limit=10,
        use_cp=use_cp,
        use_lns=use_lns,
        use_cp_sat=use_cp_sat,
    )
    solver.set_warm_start(start_solution)
    result_storage2 = solver.solve()
    print(len(result_storage2))
    print([(fit, sol.trajectories) for sol, fit in result_storage2])
    assert result_storage2[0][0].trajectories == start_solution.trajectories
