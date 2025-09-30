import pytest

from discrete_optimization.generic_scheduling_tools.solvers.lns_cp.constraint_handler import (
    ObjectiveSubproblem,
    SchedulingConstraintHandler,
)
from discrete_optimization.generic_scheduling_tools.solvers.lns_cp.neighbor_tools import (
    NeighborBuilderMix,
    NeighborBuilderSubPart,
    NeighborRandom,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.lns_cp import LnsOrtoolsCpSat
from discrete_optimization.generic_tools.lns_tools import TrivialInitialSolution
from discrete_optimization.jsp.parser import get_data_available, parse_file
from discrete_optimization.jsp.solvers.cpsat import CpSatJspSolver


@pytest.mark.parametrize("objective_subproblem", list(ObjectiveSubproblem))
def test_lns(objective_subproblem):
    problem = parse_file(get_data_available()[0])
    subsolver = CpSatJspSolver(problem=problem)
    parameters_cp = ParametersCp.default()
    initial_res = subsolver.solve(
        parameters_cp=parameters_cp, callbacks=[NbIterationStopper(nb_iteration_max=1)]
    )
    initial_solution_provider = TrivialInitialSolution(solution=initial_res)
    constraint_handler = SchedulingConstraintHandler(
        problem=problem,
        neighbor_builder=NeighborBuilderMix(
            list_neighbor=[
                NeighborBuilderSubPart(
                    problem=problem,
                ),
                NeighborRandom(problem=problem),
            ],
            weight_neighbor=[0.5, 0.5],
        ),
        objective_subproblem=objective_subproblem,
    )
    solver = LnsOrtoolsCpSat(
        problem=problem,
        subsolver=subsolver,
        constraint_handler=constraint_handler,
        initial_solution_provider=initial_solution_provider,
    )
    res = solver.solve(
        nb_iteration_lns=2,
        time_limit_subsolver=10,
        parameters_cp=parameters_cp,
    )
    sol = res.get_best_solution()
    problem.satisfy(sol)
