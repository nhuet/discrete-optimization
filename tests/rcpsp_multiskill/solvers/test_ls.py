#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


from discrete_optimization.generic_rcpsp_tools.solvers.ls import (
    LsGenericRcpspSolver,
    LsSolverType,
)
from discrete_optimization.rcpsp_multiskill.parser_imopse import (
    get_data_available,
    parse_file,
)
from discrete_optimization.rcpsp_multiskill.plots.plot_solution import (
    plot_resource_individual_gantt,
)
from discrete_optimization.rcpsp_multiskill.problem import (
    Employee,
    MultiskillRcpspSolution,
    SkillDetail,
    VariantMultiskillRcpspProblem,
)


def create_toy_msrcpsp():
    skills_set: set[str] = {"S1", "S2", "S3"}
    resources_set: set[str] = {"R1", "R2", "R3"}
    non_renewable_resources = set()
    resources_availability = {"R1": [2] * 100, "R2": [4] * 100, "R3": [3] * 100}
    employee: dict[int, Employee] = {
        1: Employee(
            dict_skill={"S1": SkillDetail(1.0, 1.0, 1.0)},
            calendar_employee=[True] * 100,
        ),
        2: Employee(
            dict_skill={"S2": SkillDetail(1.0, 1.0, 1.0)},
            calendar_employee=[True] * 100,
        ),
        3: Employee(
            dict_skill={"S3": SkillDetail(1.0, 1.0, 1.0)},
            calendar_employee=[True] * 100,
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

    model = VariantMultiskillRcpspProblem(
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
    return model


def test_ls():
    model = create_toy_msrcpsp()
    model = model.to_variant_model()
    solver = LsGenericRcpspSolver(problem=model, ls_solver=LsSolverType.SA)
    result = solver.solve(nb_iteration_max=1000)
    solution: MultiskillRcpspSolution = result.get_best_solution()
    model.evaluate(solution)
    assert model.satisfy(solution)


def test_ls_imopse():
    file = [f for f in get_data_available() if "100_5_22_15.def" in f][0]
    model, name_task = parse_file(file, max_horizon=1000)
    model = model.to_variant_model()
    solver = LsGenericRcpspSolver(problem=model, ls_solver=LsSolverType.SA)
    result = solver.solve(nb_iteration_max=1)
    solution: MultiskillRcpspSolution = result.get_best_solution()
    model.evaluate(solution)
    assert model.satisfy(solution)
    plot_resource_individual_gantt(rcpsp_problem=model, rcpsp_sol=solution)


if __name__ == "__main__":
    test_ls_imopse()
