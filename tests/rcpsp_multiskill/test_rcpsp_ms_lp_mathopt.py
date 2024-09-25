#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.generic_tools.lp_tools import MilpSolverName
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import (
    Employee,
    MS_RCPSPModel,
    SkillDetail,
)
from discrete_optimization.rcpsp_multiskill.solvers.cp_solvers import CP_MS_MRCPSP_MZN
from discrete_optimization.rcpsp_multiskill.solvers.lp_model import (
    LP_Solver_MRSCPSP,
    LP_Solver_MRSCPSP_MathOpt,
    ParametersMilp,
)


def test_lp():
    skills_set: set[str] = {"S1", "S2", "S3"}
    resources_set: set[str] = {"R1", "R2", "R3"}
    non_renewable_resources = set()
    resources_availability = {"R1": [2] * 100, "R2": [4] * 100, "R3": [3] * 100}
    employee: dict[int, Employee] = {
        1: Employee(
            dict_skill={"S1": SkillDetail(1.0, 1.0, 1.0)},
            calendar_employee=[True] * 1000,
        ),
        2: Employee(
            dict_skill={"S2": SkillDetail(1.0, 1.0, 1.0)},
            calendar_employee=[True] * 1000,
        ),
        3: Employee(
            dict_skill={"S3": SkillDetail(1.0, 1.0, 1.0)},
            calendar_employee=[True] * 1000,
        ),
    }
    employees_availability: list[int] = [3] * 1000
    mode_details: dict[int, dict[int, dict[str, int]]] = {
        1: {1: {"R1": 0, "R2": 0, "R3": 0, "duration": 0}},
        2: {1: {"S1": 1, "R1": 2, "R2": 0, "R3": 0, "duration": 2}},
        3: {1: {"S2": 1, "R1": 1, "R2": 2, "R3": 0, "duration": 4}},
        4: {1: {"S3": 1, "R1": 2, "R2": 0, "R3": 0, "duration": 5}},
        5: {1: {"R1": 0, "R2": 0, "R3": 0, "duration": 0}},
    }
    successors: dict[int, list[int]] = {1: [2, 3], 2: [5], 3: [4], 4: [5], 5: []}

    model = MS_RCPSPModel(
        skills_set=skills_set,
        resources_set=resources_set,
        non_renewable_resources=non_renewable_resources,
        resources_availability=resources_availability,
        employees=employee,
        employees_availability=employees_availability,
        mode_details=mode_details,
        successors=successors,
        horizon=100,
        horizon_multiplier=1,
    )
    lp_solver = LP_Solver_MRSCPSP_MathOpt(problem=model)
    lp_solver.init_model()

    res = lp_solver.solve(parameters_milp=ParametersMilp.default())
    sol = res.get_best_solution()
    assert model.satisfy(sol)

    # test warm start
    # start solution
    start_solver = LP_Solver_MRSCPSP(problem=model, lp_solver=MilpSolverName.CBC)
    start_solution = start_solver.solve(time_limit=10).get_best_solution()
    assert model.satisfy(start_solution)

    # check different from first solution found
    assert (
        res[0][0].schedule != start_solution.schedule
        or res[0][0].modes != start_solution.modes
        or res[0][0].employee_usage != start_solution.employee_usage
    )

    # solve with warm_start
    lp_solver.set_warm_start(start_solution)
    res2 = lp_solver.solve(parameters_milp=ParametersMilp.default())

    # check first solution is the warmstart
    assert (
        res2[0][0].schedule == start_solution.schedule
        and res2[0][0].modes == start_solution.modes
        and res2[0][0].employee_usage == start_solution.employee_usage
    )


def test_lp_bis():
    skills_set: set[str] = {"S1", "S2", "S3"}
    resources_set: set[str] = {"R1", "R2", "R3"}
    non_renewable_resources = set()
    resources_availability = {"R1": [2] * 100, "R2": [4] * 100, "R3": [3] * 100}
    employee: dict[int, Employee] = {
        1: Employee(
            dict_skill={"S1": SkillDetail(1.0, 1.0, 1.0)},
            calendar_employee=[True] * 1000,
        ),
        2: Employee(
            dict_skill={"S2": SkillDetail(1.0, 1.0, 1.0)},
            calendar_employee=[True] * 1000,
        ),
        3: Employee(
            dict_skill={"S3": SkillDetail(1.0, 1.0, 1.0)},
            calendar_employee=[True] * 1000,
        ),
    }
    index = 5
    for emp in sorted(employee):
        indexes = [index + 8 * i for i in range(10)] + [
            index + 1 + 8 * i for i in range(10)
        ]
        for i in indexes:
            employee[emp].calendar_employee[i] = False
        index += 1

    employees_availability: list[int] = [3] * 1000
    mode_details: dict[int, dict[int, dict[str, int]]] = {
        1: {1: {"R1": 0, "R2": 0, "R3": 0, "duration": 0}},
        2: {
            1: {"S1": 1, "R1": 2, "R2": 0, "R3": 0, "duration": 2},
            2: {"S2": 1, "R1": 0, "R2": 0, "R3": 0, "duration": 3},
        },
        3: {1: {"S2": 1, "R1": 1, "R2": 2, "R3": 0, "duration": 4}},
        4: {1: {"S3": 1, "R1": 2, "R2": 0, "R3": 0, "duration": 5}},
        5: {1: {"R1": 2, "R2": 0, "R3": 0, "duration": 5}},
        6: {1: {"S3": 1, "S2": 1, "R1": 2, "R2": 0, "R3": 0, "duration": 5}},
        7: {
            1: {"S3": 1, "R1": 2, "R2": 0, "R3": 0, "duration": 1},
            2: {"R1": 2, "R2": 0, "R3": 0, "duration": 2},
        },
        8: {1: {"R1": 0, "R2": 0, "R3": 0, "duration": 0}},
    }
    successors: dict[int, list[int]] = {
        1: [2, 3],
        2: [5],
        3: [4],
        4: [5],
        5: [6, 7],
        6: [8],
        7: [8],
        8: [],
    }

    model = MS_RCPSPModel(
        skills_set=skills_set,
        resources_set=resources_set,
        non_renewable_resources=non_renewable_resources,
        resources_availability=resources_availability,
        employees=employee,
        employees_availability=employees_availability,
        mode_details=mode_details,
        successors=successors,
        horizon=100,
        horizon_multiplier=1,
    )
    lp_solver = LP_Solver_MRSCPSP_MathOpt(problem=model)
    lp_solver.init_model()
    result = lp_solver.solve(parameters_milp=ParametersMilp.default())
    best_solution = result.get_best_solution()
    model.evaluate(best_solution)
    assert model.satisfy(best_solution)