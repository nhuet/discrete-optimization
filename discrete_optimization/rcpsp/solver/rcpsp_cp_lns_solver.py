import random
from enum import Enum
from typing import Any, Iterable, List, Optional, Union

import numpy as np
from minizinc import Instance

from discrete_optimization.generic_tools.cp_tools import CPSolverName, ParametersCP
from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.generic_tools.lns_cp import (
    LNS_CP,
    ConstraintHandler,
    SolverDO,
)
from discrete_optimization.generic_tools.lns_mip import PostProcessSolution
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.rcpsp_model import (
    PartialSolution,
    RCPSPModel,
    RCPSPSolution,
)
from discrete_optimization.rcpsp.solver.cp_solvers import (
    CP_MRCPSP_MZN,
    CP_MRCPSP_MZN_WITH_FAKE_TASK,
    CP_RCPSP_MZN,
    CPSolver,
)
from discrete_optimization.rcpsp.solver.ls_solver import LS_SOLVER, LS_RCPSP_Solver
from discrete_optimization.rcpsp.solver.rcpsp_lp_lns_solver import (
    InitialMethodRCPSP,
    InitialSolutionRCPSP,
)


# Strategy n°1 : fixing subset of starting time
class ConstraintHandlerFixStartingTime(ConstraintHandler):
    def __init__(
        self,
        problem: RCPSPModel,
        fraction_to_fix: float = 0.9,
        delta_time_from_makepan_to_not_fix: int = 5,
    ):
        self.problem = problem
        self.fraction_to_fix = fraction_to_fix
        self.delta_time_from_makepan_to_not_fix = delta_time_from_makepan_to_not_fix

    def adding_constraint_from_results_store(
        self,
        cp_solver: Union[CP_RCPSP_MZN, CP_MRCPSP_MZN],
        child_instance: Instance,
        result_storage: ResultStorage,
        last_result_store: Optional[ResultStorage] = None,
    ) -> Iterable[Any]:
        current_solution, fit = result_storage.get_best_solution_fit()
        max_time = max(
            [
                current_solution.rcpsp_schedule[x]["end_time"]
                for x in current_solution.rcpsp_schedule
            ]
        )
        last_jobs = [
            x
            for x in current_solution.rcpsp_schedule
            if current_solution.rcpsp_schedule[x]["end_time"]
            >= max_time - self.delta_time_from_makepan_to_not_fix
        ]
        nb_jobs = self.problem.n_jobs
        jobs_to_fix = set(
            random.sample(
                current_solution.rcpsp_schedule.keys(),
                int(self.fraction_to_fix * nb_jobs),
            )
        )
        for lj in last_jobs:
            if lj in jobs_to_fix:
                jobs_to_fix.remove(lj)
        list_strings = []
        for job in jobs_to_fix:
            start_time_j = current_solution.rcpsp_schedule[job]["start_time"]
            if isinstance(cp_solver, CP_RCPSP_MZN):
                string1 = (
                    "constraint s["
                    + str(cp_solver.index_in_minizinc[job])
                    + "]=="
                    + str(start_time_j)
                    + ";\n"
                )
            elif isinstance(cp_solver, CP_MRCPSP_MZN):
                string1 = (
                    "constraint start["
                    + str(cp_solver.index_in_minizinc[job])
                    + "]=="
                    + str(start_time_j)
                    + ";\n"
                )
            list_strings += [string1]
            child_instance.add_string(string1)
        for job in current_solution.rcpsp_schedule:
            if isinstance(cp_solver, CP_RCPSP_MZN):
                string = (
                    "constraint s["
                    + str(cp_solver.index_in_minizinc[job])
                    + "] <= "
                    + str(max_time)
                    + ";\n"
                )
            if isinstance(cp_solver, CP_MRCPSP_MZN):
                string = (
                    "constraint start["
                    + str(cp_solver.index_in_minizinc[job])
                    + "] <= "
                    + str(max_time)
                    + ";\n"
                )
            child_instance.add_string(string)
        return list_strings

    def remove_constraints_from_previous_iteration(
        self,
        cp_solver: CP_RCPSP_MZN,
        child_instance,
        previous_constraints: Iterable[Any],
    ):
        pass


class ConstraintHandlerStartTimeInterval_CP(ConstraintHandler):
    def __init__(
        self,
        problem: RCPSPModel,
        fraction_to_fix: float = 0.9,
        minus_delta: int = 2,
        plus_delta: int = 2,
        delta_time_from_makepan_to_not_fix: int = 5,
    ):
        self.problem = problem
        self.fraction_to_fix = fraction_to_fix
        self.minus_delta = minus_delta
        self.plus_delta = plus_delta
        self.delta_time_from_makepan_to_not_fix = delta_time_from_makepan_to_not_fix

    def adding_constraint_from_results_store(
        self,
        cp_solver: Union[CP_RCPSP_MZN, CP_MRCPSP_MZN],
        child_instance: Instance,
        result_storage: ResultStorage,
        last_result_store: Optional[ResultStorage] = None,
    ) -> Iterable[Any]:
        current_solution, fit = result_storage.get_best_solution_fit()
        max_time = max(
            [
                current_solution.rcpsp_schedule[x]["end_time"]
                for x in current_solution.rcpsp_schedule
            ]
        )
        last_jobs = [
            x
            for x in current_solution.rcpsp_schedule
            if current_solution.rcpsp_schedule[x]["end_time"]
            >= max_time - self.delta_time_from_makepan_to_not_fix
        ]
        nb_jobs = self.problem.n_jobs
        jobs_to_fix = set(
            random.sample(
                current_solution.rcpsp_schedule.keys(),
                int(self.fraction_to_fix * nb_jobs),
            )
        )
        for lj in last_jobs:
            if lj in jobs_to_fix:
                jobs_to_fix.remove(lj)
        list_strings = []
        modes_dict = self.problem.build_mode_dict(current_solution.rcpsp_modes)
        for t in modes_dict:
            if isinstance(cp_solver, CP_MRCPSP_MZN):
                list_strings += cp_solver.constraint_task_to_mode(
                    task_id=t, mode=modes_dict[t]
                )
        for job in jobs_to_fix:
            start_time_j = current_solution.rcpsp_schedule[job]["start_time"]
            min_st = max(start_time_j - self.minus_delta, 0)
            max_st = min(start_time_j + self.plus_delta, max_time)
            if isinstance(cp_solver, CP_RCPSP_MZN):
                string1 = (
                    "constraint s["
                    + str(cp_solver.index_in_minizinc[job])
                    + "] <= "
                    + str(max_st)
                    + ";\n"
                )
                string2 = (
                    "constraint s["
                    + str(cp_solver.index_in_minizinc[job])
                    + "] >= "
                    + str(min_st)
                    + ";\n"
                )
            elif isinstance(cp_solver, CP_MRCPSP_MZN):
                string1 = (
                    "constraint start["
                    + str(cp_solver.index_in_minizinc[job])
                    + "] <= "
                    + str(max_st)
                    + ";\n"
                )
                string2 = (
                    "constraint start["
                    + str(cp_solver.index_in_minizinc[job])
                    + "] >= "
                    + str(min_st)
                    + ";\n"
                )
                list_strings += cp_solver.constraint_task_to_mode(
                    task_id=job, mode=modes_dict[job]
                )
            list_strings += [string1]
            list_strings += [string2]
            child_instance.add_string(string1)
            child_instance.add_string(string2)
        return list_strings

    def remove_constraints_from_previous_iteration(
        self,
        cp_solver: CP_RCPSP_MZN,
        child_instance,
        previous_constraints: Iterable[Any],
    ):
        pass


class ConstraintHandlerByPart_CP(ConstraintHandler):
    def __init__(
        self,
        problem: RCPSPModel,
        fraction_to_fix: float = 0.9,
        nb_cut_part: int = 10,
        minus_delta: int = 2,
        plus_delta: int = 2,
        delta_time_from_makepan_to_not_fix: int = 5,
    ):
        self.fraction_to_fix = fraction_to_fix
        self.problem = problem
        self.nb_cut_part = nb_cut_part
        self.minus_delta = minus_delta
        self.plus_delta = plus_delta
        self.delta_time_from_makepan_to_not_fix = delta_time_from_makepan_to_not_fix
        self.current_sub_part = 0

    def adding_constraint_from_results_store(
        self,
        cp_solver: Union[CP_RCPSP_MZN, CP_MRCPSP_MZN],
        child_instance: Instance,
        result_storage: ResultStorage,
        last_result_store: Optional[ResultStorage] = None,
    ) -> Iterable[Any]:
        current_solution, fit = result_storage.get_best_solution_fit()
        max_time = max(
            [
                current_solution.rcpsp_schedule[x]["end_time"]
                for x in current_solution.rcpsp_schedule
            ]
        )
        delta_t = max_time / self.nb_cut_part
        task_of_interest = [
            t
            for t in current_solution.rcpsp_schedule
            if delta_t * self.current_sub_part
            <= current_solution.rcpsp_schedule[t]["start_time"]
            <= delta_t * (self.current_sub_part + 1)
        ]
        last_jobs = [
            x
            for x in current_solution.rcpsp_schedule
            if current_solution.rcpsp_schedule[x]["end_time"]
            >= max_time - self.delta_time_from_makepan_to_not_fix
        ]
        nb_jobs = self.problem.n_jobs
        jobs_to_fix = set(
            random.sample(
                current_solution.rcpsp_schedule.keys(),
                int(self.fraction_to_fix * nb_jobs),
            )
        )
        for lj in last_jobs:
            if lj in jobs_to_fix:
                jobs_to_fix.remove(lj)
        for t in task_of_interest:
            if t in jobs_to_fix:
                jobs_to_fix.remove(t)
        list_strings = []
        for job in jobs_to_fix:
            start_time_j = current_solution.rcpsp_schedule[job]["start_time"]
            min_st = max(start_time_j - self.minus_delta, 0)
            max_st = min(start_time_j + self.plus_delta, max_time)
            if isinstance(cp_solver, CP_RCPSP_MZN):
                string1 = "constraint s[" + str(job) + "] <= " + str(max_st) + ";\n"
                string2 = "constraint s[" + str(job) + "] >= " + str(min_st) + ";\n"
            elif isinstance(cp_solver, CP_MRCPSP_MZN):
                string1 = "constraint start[" + str(job) + "] <= " + str(max_st) + ";\n"
                string2 = "constraint start[" + str(job) + "] >= " + str(min_st) + ";\n"
            list_strings += [string1]
            list_strings += [string2]
            child_instance.add_string(string1)
            child_instance.add_string(string2)
        for job in current_solution.rcpsp_schedule:
            if isinstance(cp_solver, CP_RCPSP_MZN):
                string = (
                    "constraint s[" + str(job) + "] <= " + str(max_time + 50) + ";\n"
                )
            if isinstance(cp_solver, CP_MRCPSP_MZN):
                string = (
                    "constraint start["
                    + str(job)
                    + "] <= "
                    + str(max_time + 50)
                    + ";\n"
                )
            child_instance.add_string(string)
        self.current_sub_part = (self.current_sub_part + 1) % self.nb_cut_part
        return list_strings

    def remove_constraints_from_previous_iteration(
        self,
        cp_solver: CP_RCPSP_MZN,
        child_instance,
        previous_constraints: Iterable[Any],
    ):
        pass


class OptionNeighbor(Enum):
    MIX_ALL = 0
    MIX_FAST = 1
    MIX_LARGE_NEIGH = 2
    LARGE = 4
    DEBUG = 3


class Params:
    fraction_to_fix: float
    minus_delta: int
    plus_delta: int
    delta_time_from_makepan_to_not_fix: int

    def __init__(
        self,
        fraction_to_fix: float = 0.9,
        minus_delta: int = 2,
        plus_delta: int = 2,
        delta_time_from_makepan_to_not_fix: int = 5,
    ):
        self.fraction_to_fix = fraction_to_fix
        self.minus_delta = minus_delta
        self.plus_delta = plus_delta
        self.delta_time_from_makepan_to_not_fix = delta_time_from_makepan_to_not_fix


class ConstraintHandlerMix(ConstraintHandler):
    def __init__(
        self, problem: RCPSPModel, list_params: List[Params], list_proba: List[float]
    ):
        self.problem = problem
        self.list_params = list_params
        self.list_proba = list_proba
        if isinstance(self.list_proba, list):
            self.list_proba = np.array(self.list_proba)
        self.list_proba = self.list_proba / np.sum(self.list_proba)
        self.index_np = np.array(range(len(self.list_proba)), dtype=np.int)
        self.current_iteration = 0
        self.status = {
            i: {"nb_usage": 0, "nb_improvement": 0}
            for i in range(len(self.list_params))
        }
        self.last_index_param = None
        self.last_fitness = None

    def adding_constraint_from_results_store(
        self,
        cp_solver: CP_MRCPSP_MZN,
        child_instance: Instance,
        result_storage: ResultStorage,
        last_result_store: Optional[ResultStorage] = None,
    ) -> Iterable[Any]:
        new_fitness = result_storage.get_best_solution_fit()[1]
        if self.last_index_param is not None:
            if new_fitness != self.last_fitness:
                self.status[self.last_index_param]["nb_improvement"] += 1
                self.last_fitness = new_fitness
                self.list_proba[self.last_index_param] *= 1.05
                self.list_proba = self.list_proba / np.sum(self.list_proba)
            else:
                self.list_proba[self.last_index_param] *= 0.95
                self.list_proba = self.list_proba / np.sum(self.list_proba)
        else:
            self.last_fitness = new_fitness
        if random.random() <= 0.95:
            choice = np.random.choice(self.index_np, size=1, p=self.list_proba)[0]
        else:
            max_improvement = max(
                [
                    self.status[x]["nb_improvement"]
                    / max(self.status[x]["nb_usage"], 1)
                    for x in self.status
                ]
            )
            choice = random.choice(
                [
                    x
                    for x in self.status
                    if self.status[x]["nb_improvement"]
                    / max(self.status[x]["nb_usage"], 1)
                    == max_improvement
                ]
            )
        d_params = {
            key: getattr(self.list_params[int(choice)], key)
            for key in self.list_params[0].__dict__.keys()
        }
        ch = ConstraintHandlerStartTimeInterval_CP(problem=self.problem, **d_params)
        self.current_iteration += 1
        self.last_index_param = choice
        self.status[self.last_index_param]["nb_usage"] += 1
        return ch.adding_constraint_from_results_store(
            cp_solver, child_instance, result_storage, last_result_store
        )

    def remove_constraints_from_previous_iteration(
        self, cp_solver: CPSolver, child_instance, previous_constraints: Iterable[Any]
    ):
        pass


class PostProcessLeftShift(PostProcessSolution):
    def __init__(
        self, rcpsp_problem: RCPSPModel, partial_solution: PartialSolution = None
    ):
        self.rcpsp_problem = rcpsp_problem
        self.partial_solution = partial_solution
        if self.partial_solution is None:

            def check_solution(problem, solution):
                return True

        else:

            def check_solution(problem, solution):
                start_together = partial_solution.start_together
                start_at_end = partial_solution.start_at_end
                start_at_end_plus_offset = partial_solution.start_at_end_plus_offset
                start_after_nunit = partial_solution.start_after_nunit
                for (t1, t2) in start_together:
                    b = (
                        solution.rcpsp_schedule[t1]["start_time"]
                        == solution.rcpsp_schedule[t2]["start_time"]
                    )
                    if not b:
                        return False
                for (t1, t2) in start_at_end:
                    b = (
                        solution.rcpsp_schedule[t2]["start_time"]
                        == solution.rcpsp_schedule[t1]["end_time"]
                    )
                    if not b:
                        return False
                for (t1, t2, off) in start_at_end_plus_offset:
                    b = (
                        solution.rcpsp_schedule[t2]["start_time"]
                        >= solution.rcpsp_schedule[t1]["end_time"] + off
                    )
                    if not b:
                        return False
                for (t1, t2, off) in start_after_nunit:
                    b = (
                        solution.rcpsp_schedule[t2]["start_time"]
                        >= solution.rcpsp_schedule[t1]["start_time"] + off
                    )
                    if not b:
                        return False
                return True

        self.check_sol = check_solution

    def build_other_solution(self, result_storage: ResultStorage) -> ResultStorage:
        for sol in list(result_storage.list_solution_fits):
            if "satisfy" not in sol[0].__dict__.keys():
                s: RCPSPSolution = sol[0]
                sol[0].satisfy = self.check_sol(
                    problem=self.rcpsp_problem, solution=s
                ) and self.rcpsp_problem.satisfy(s)
            if self.partial_solution is None:
                s: RCPSPSolution = sol[0]
                solution = RCPSPSolution(
                    problem=self.rcpsp_problem,
                    rcpsp_permutation=s.rcpsp_permutation,
                    rcpsp_modes=s.rcpsp_modes,
                )
                solution.satisfy = self.check_sol(
                    problem=self.rcpsp_problem, solution=solution
                ) and self.rcpsp_problem.satisfy(solution)
                result_storage.list_solution_fits += [
                    (solution, -self.rcpsp_problem.evaluate(solution)["makespan"])
                ]
        if self.partial_solution is None:
            solver = LS_RCPSP_Solver(model=self.rcpsp_problem, ls_solver=LS_SOLVER.SA)
            satisfiable = [
                (s, f) for s, f in result_storage.list_solution_fits if s.satisfy
            ]
            if len(satisfiable) > 0:
                s: RCPSPSolution = max(satisfiable, key=lambda x: x[1])[0].copy()
            else:
                s = result_storage.get_best_solution().copy()
            s.change_problem(self.rcpsp_problem)
            result_store = solver.solve(nb_iteration_max=50, init_solution=s)
            for solution, f in result_store.list_solution_fits:
                solution.satisfy = self.check_sol(self.rcpsp_problem, solution)
                result_storage.list_solution_fits += [
                    (solution, -self.rcpsp_problem.evaluate(solution)["makespan"])
                ]
        return result_storage


def build_neighbor_operator(option_neighbor: OptionNeighbor, rcpsp_model):
    params_om = [Params(fraction_to_fix=0.75, minus_delta=100, plus_delta=100)]
    params_all = [
        Params(fraction_to_fix=0.9, minus_delta=1, plus_delta=1),
        Params(fraction_to_fix=0.85, minus_delta=3, plus_delta=3),
        Params(fraction_to_fix=0.9, minus_delta=4, plus_delta=4),
        Params(fraction_to_fix=0.9, minus_delta=4, plus_delta=4),
        Params(fraction_to_fix=0.92, minus_delta=10, plus_delta=0),
        Params(fraction_to_fix=0.88, minus_delta=0, plus_delta=10),
        Params(fraction_to_fix=0.9, minus_delta=10, plus_delta=0),
        Params(fraction_to_fix=0.8, minus_delta=5, plus_delta=5),
        Params(fraction_to_fix=0.85, minus_delta=15, plus_delta=15),
        Params(fraction_to_fix=0.9, minus_delta=3, plus_delta=3),
        Params(fraction_to_fix=1.0, minus_delta=5, plus_delta=5),
        Params(fraction_to_fix=0.85, minus_delta=1, plus_delta=1),
        Params(fraction_to_fix=0.8, minus_delta=2, plus_delta=2),
        Params(fraction_to_fix=0.85, minus_delta=5, plus_delta=5),
        Params(fraction_to_fix=0.85, minus_delta=5, plus_delta=5),
        Params(fraction_to_fix=0.85, minus_delta=5, plus_delta=5),
        Params(fraction_to_fix=0.85, minus_delta=5, plus_delta=5),
        Params(fraction_to_fix=0.95, minus_delta=5, plus_delta=5),
        Params(fraction_to_fix=0.95, minus_delta=5, plus_delta=5),
        Params(fraction_to_fix=0.85, minus_delta=5, plus_delta=5),
        Params(fraction_to_fix=0.9, minus_delta=1, plus_delta=1),
        Params(fraction_to_fix=0.9, minus_delta=1, plus_delta=1),
        Params(fraction_to_fix=0.8, minus_delta=2, plus_delta=2),
        Params(fraction_to_fix=0.98, minus_delta=2, plus_delta=2),
        Params(fraction_to_fix=0.9, minus_delta=3, plus_delta=3),
        Params(fraction_to_fix=0.98, minus_delta=3, plus_delta=3),
        Params(fraction_to_fix=0.98, minus_delta=8, plus_delta=8),
        Params(fraction_to_fix=0.98, minus_delta=10, plus_delta=10),
    ]
    params_fast = [
        Params(fraction_to_fix=0.9, minus_delta=1, plus_delta=1),
        Params(fraction_to_fix=0.8, minus_delta=1, plus_delta=1),
        Params(fraction_to_fix=0.8, minus_delta=2, plus_delta=2),
        Params(fraction_to_fix=0.9, minus_delta=1, plus_delta=1),
        Params(fraction_to_fix=0.92, minus_delta=3, plus_delta=3),
        Params(fraction_to_fix=0.98, minus_delta=7, plus_delta=7),
        Params(fraction_to_fix=0.95, minus_delta=5, plus_delta=5),
    ]
    params_debug = [Params(fraction_to_fix=1.0, minus_delta=0, plus_delta=0)]
    params_large = [
        Params(fraction_to_fix=0.9, minus_delta=12, plus_delta=12),
        Params(fraction_to_fix=0.8, minus_delta=3, plus_delta=3),
        Params(fraction_to_fix=0.7, minus_delta=12, plus_delta=12),
        Params(fraction_to_fix=0.7, minus_delta=5, plus_delta=5),
        Params(fraction_to_fix=0.6, minus_delta=3, plus_delta=3),
        Params(fraction_to_fix=0.4, minus_delta=2, plus_delta=2),
        Params(fraction_to_fix=0.9, minus_delta=4, plus_delta=4),
        Params(fraction_to_fix=0.7, minus_delta=4, plus_delta=4),
        Params(fraction_to_fix=0.8, minus_delta=5, plus_delta=5),
    ]
    params = None
    if option_neighbor == OptionNeighbor.MIX_ALL:
        params = params_all
    if option_neighbor == OptionNeighbor.MIX_FAST:
        params = params_fast
    if option_neighbor == OptionNeighbor.MIX_LARGE_NEIGH:
        params = params_large
    if option_neighbor == OptionNeighbor.DEBUG:
        params = params_debug
    if option_neighbor == OptionNeighbor.LARGE:
        params = params_om
    probas = [1 / len(params)] * len(params)
    constraint_handler = ConstraintHandlerMix(
        problem=rcpsp_model, list_params=params, list_proba=probas
    )
    return constraint_handler


class LNS_CP_RCPSP_SOLVER(SolverDO):
    def __init__(
        self,
        rcpsp_model: RCPSPModel,
        option_neighbor: OptionNeighbor = OptionNeighbor.MIX_ALL,
        **kwargs
    ):
        self.rcpsp_model = rcpsp_model
        self.with_varying_resource = rcpsp_model.is_varying_resource()
        if self.with_varying_resource:
            self.solver = CP_MRCPSP_MZN_WITH_FAKE_TASK(
                rcpsp_model=self.rcpsp_model, cp_solver_name=CPSolverName.CHUFFED
            )
        else:
            self.solver = CP_MRCPSP_MZN(
                rcpsp_model=self.rcpsp_model, cp_solver_name=CPSolverName.CHUFFED
            )
        self.solver.init_model(output_type=True, **kwargs)
        self.parameters_cp = ParametersCP.default()
        params_objective_function = get_default_objective_setup(
            problem=self.rcpsp_model
        )
        self.constraint_handler = build_neighbor_operator(
            option_neighbor=option_neighbor, rcpsp_model=self.rcpsp_model
        )
        self.initial_solution_provider = InitialSolutionRCPSP(
            problem=self.rcpsp_model,
            initial_method=InitialMethodRCPSP.DUMMY,
            params_objective_function=params_objective_function,
        )
        self.lns_solver = LNS_CP(
            problem=self.rcpsp_model,
            cp_solver=self.solver,
            post_process_solution=PostProcessLeftShift(
                self.rcpsp_model, partial_solution=kwargs.get("partial_solution", None)
            ),
            initial_solution_provider=self.initial_solution_provider,
            constraint_handler=self.constraint_handler,
            params_objective_function=params_objective_function,
        )

    def solve(self, **kwargs) -> ResultStorage:
        return self.lns_solver.solve_lns(
            parameters_cp=kwargs.get("parameters_cp", self.parameters_cp),
            max_time_seconds=kwargs.get("max_time_seconds", 1000),
            skip_first_iteration=kwargs.get("skip_first_iteration", False),
            nb_iteration_no_improvement=kwargs.get("nb_iteration_no_improvement", 100),
            nb_iteration_lns=kwargs.get("nb_iteration_lns", 100),
        )
