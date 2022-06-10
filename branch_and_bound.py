
import numpy as np
import sys
import time
from collections import deque

import mip
from mip import Model, CutType, OptimizationStatus
import os

from utils import create_model
from choose_cut import generate_cut, choose_cut

from DataGenerator.SetCover import generate_setcover
from DataGenerator.TravelingSalesman import generate_traveling_salesman
from DataGenerator.GeneralMIP import generate_general_mip
from DataGenerator.BoundedKnapsack import generate_bounded_knapsack
from DataGenerator.FacilityLocation import generate_facility_location

def _find_branch_var(sol, integral_list, thresh=1e-6):
    """Find the branching variable with given LP solution.

    :returns: Index for branching variable. Returns none if the solution already satisfies the 
              integrality condition.
    :rtype: int or None
    """
    masked_sol = sol * integral_list if integral_list is not None else sol
    fractional_part = np.abs(np.round(masked_sol, 6) - np.round(masked_sol))
    if np.all(fractional_part < thresh):
        return None
    else:
        return np.argmax(fractional_part)


def _relax_and_solve(model):
    model.verbose = 0
    status = model.optimize(relax=True)
    if status == mip.OptimizationStatus.NO_SOLUTION_FOUND:
        return None, np.inf
    elif status == mip.OptimizationStatus.INFEASIBLE or status == mip.OptimizationStatus.UNBOUNDED:
        return None, np.inf
    elif status == mip.OptimizationStatus.OPTIMAL or status == mip.OptimizationStatus.FEASIBLE:
        obj = model.objective_value
        sol = np.array([v.x for v in model.vars])
        return sol, obj


def branch_and_bound(model, x, integral_list, sense="MIN", time_limit=30, verbose=0):
    """

    By defalut minimizing the objective.
    """

    start_time = time.time()
    time_elapsed = 0
    logging_interval = 1  # every 1 sec print 1 row of logs
    logging_cnt = 0

    model.verbose = 0
    node_queue = deque()
    node_queue.append(model)
    node_cnt = 0
    best_obj = np.inf
    best_sol = None

    # sense = 1, minimize; sense = -1, maximize
    sense_coeff = 1 if sense == "MIN" else -1
    best_obj *= sense_coeff

    while node_queue and time_elapsed < time_limit:
        ''' randomly sample node '''
        # node_idx = np.random.choice([i for i in range(len(node_queue))])
        # node = node_queue[node_idx]
        # node_queue.remove(node)
        ''' FIFO '''
        node = node_queue.pop()

        t0 = time.time()
        lp_sol, lp_obj = _relax_and_solve(node)
        t1 = time.time()
        if verbose >= 3:
            print(f"\tVisiting {node_cnt}-th node, LP solving time is {t1 - t0}")

        node_cnt += 1

        if lp_sol is None: continue  # Infeasible
        elif sense_coeff * (lp_obj - best_obj) > 0: continue  # Cutoff

        branch_idx = _find_branch_var(lp_sol, integral_list)
        if branch_idx is None:
            if (best_obj - lp_obj) * sense_coeff > 0:
                best_obj = lp_obj
                best_sol = lp_sol
        else:
            if verbose >= 3:
                print(f"\tBranching on variable x_{branch_idx}")
            branch_l = np.floor(lp_sol[branch_idx])
            node_l = node.copy()
            node_l += x[branch_idx] <= branch_l
            node_queue.append(node_l)

            branch_r = branch_l + 1 
            node_r = node.copy()
            node_r += x[branch_idx] >= branch_r
            node_queue.append(node_r)
        
        time_elapsed = time.time() - start_time

        if verbose >= 2 and time_elapsed > logging_interval * logging_cnt:
            logging_cnt += 1
            print(
                "{:>5d} nodes solved, {:>4d} nodes on tree, {:>6.2f} s elapsed from start".format(
                    node_cnt, len(node_queue), time_elapsed
                )
            )
    if verbose >= 1:
        print(
            "{:>5d} nodes solved, {:>4d} nodes on tree, {:>6.2f} s elapsed from start".format(
                node_cnt, len(node_queue), time_elapsed
            )
        )

    time_exceeded = (time_elapsed >= time_limit)
    
    if time_exceeded and verbose >= 1:
        if best_sol is None:
            print(
                "Time limit {:.2f} s exceeded!\
                    \nUnable to fine any feasible solution.".format(
                    time_limit
                )
            )
        else:
            print(
                "Time limit {:.2f} s exceeded!\
                    \nCurrent best solution found with objective {}.".format(
                    time_limit, best_obj
                )
            )
    elif verbose >= 1:
        if best_sol is None:
            print(
                "Model is infeasible!\
                    \nSolution completed in {:.2f} s".format(
                    time_elapsed
                )
            )
        else:
            print(
                "Optimal solution found with objective {}.\
                    \nSolution completed in {:.2f} s".format(
                    best_obj, time_elapsed
                )
            )

    return best_obj, best_sol, time_exceeded


def solve(**kwargs):
    name = kwargs.get("name")
    integral_list = kwargs.get("integral_list")
    sense = kwargs.get("sense", mip.MINIMIZE)
    A = kwargs.get("A")
    b = kwargs.get("b")
    c = kwargs.get("c")
    verbose = kwargs.get("verbose")

    # Solve the problem
    start_time = time.time()
    m, x, constr = create_model(name, A, b, c, integral_list, sense)
    obj, sol, time_exceeded = branch_and_bound(m, x, integral_list, sense, verbose=verbose)

    # Print the solution
    if sol is not None:
        if verbose >= 2:
            print("Solution:")
            print(sol)
    end_time = time.time()
    if verbose >= 1:
        print("\nSolution complete, total time: {:.3f} s".format(end_time - start_time))
    
    return obj, sol, time_exceeded


def solve_with_cut(**kwargs):
    name = kwargs.get("name")
    integral_list = kwargs.get("integral_list")
    sense = kwargs.get("sense", mip.MINIMIZE)
    A = kwargs.get("A")
    b = kwargs.get("b")
    c = kwargs.get("c")
    cut_choice_method = kwargs.get("cut_choice_method")
    cut_percentage = kwargs.get("cut_percentage")
    verbose = kwargs.get("verbose")

    # Initialize model
    succeed_flag = True
    start_time = time.time()
    m, x, constr = create_model(name, A, b, c, integral_list, sense)

    # Generate and choose cuts
    cp, relax_sol = generate_cut(m)
    if len(cp.cuts) <= 0:
        succeed_flag = False
        return succeed_flag, []

    start_cuts_time = time.time()
    selected_cuts, bag_features = choose_cut(
        cp, name=name, method=cut_choice_method, A=A, c=c, x=x,
        relax_sol=relax_sol, model=m, ratio=cut_percentage, verbose=verbose
    )
    if selected_cuts == []:
        succeed_flag = False
        return succeed_flag, bag_features
    end_cuts_time = time.time()
    if verbose >= 1:
        print(f"Time for generating cuts and collect cut features is "
              f"{end_cuts_time - start_cuts_time:.3f} s")

    # Solve the problem with cut
    m.relax()
    for cut in selected_cuts:
        m += cut
    obj, sol, time_exceeded = branch_and_bound(m, x, integral_list, sense, verbose=verbose)

    # Print the solution
    if sol is not None:
        if verbose >= 2:
            print("Solution:")
            print(sol)
    end_time = time.time()
    if verbose >= 1:
        print(f"Solution complete, total time: {end_time - start_time:.3f} s")
    if time_exceeded:
        succeed_flag = False
    return succeed_flag, bag_features


def solve_mip(**kwargs):
    name = kwargs.get("name")
    integral_list = kwargs.get("integral_list")
    sense = kwargs.get("sense", mip.MINIMIZE)
    A = kwargs.get("A")
    b = kwargs.get("b")
    c = kwargs.get("c")
    verbose = kwargs.get("verbose")

    # Solve the problem
    start_time = time.time()
    m, x, constr = create_model(name, A, b, c, integral_list, sense)
    m.verbose = verbose
    status = m.optimize(max_seconds=120)

    # Print the solution
    if status == mip.OptimizationStatus.OPTIMAL:
        if verbose >= 1:
            print(f"Optimal solution cost {m.objective_value} found")
    elif status == mip.OptimizationStatus.FEASIBLE:
        if verbose >= 1:
            print(f"Sol.cost {m.objective_value} found, best possible: {m.objective_bound}")
    elif status == mip.OptimizationStatus.NO_SOLUTION_FOUND:
        if verbose >= 1:
            print(f"No feasible solution found, lower bound is: {m.objective_bound}")
    if status == mip.OptimizationStatus.OPTIMAL or status == mip.OptimizationStatus.FEASIBLE:
        sol = np.array([v.x for v in m.vars])
        if verbose >= 1:
            print("Solution:\n\t", end="")
            print(sol)
    else:
        sol = None

    end_time = time.time()
    if verbose >= 1:
        print("\nSolution complete, total time: {:.3f} s".format(end_time - start_time))
    
    return sol, m.objective_value


def main():
    problem_names = ["SetCover", "TravelingSalesman", "GeneralMIP", "Knapsack", "FacilityLocation"]
    n_features = 14
    dens = 0.3
    max_coef = 10
    total_features = np.array([])
    total_labels = []
    cut_choice_method = "CollectLabels"
    cut_percentage = 0.3
    n_steps = 10
    
    for problem_idx in range(len(problem_names)):
        if problem_idx in [0,1,2,3]: continue

        name = problem_names[problem_idx]
        rng = np.random.RandomState(2)
        total_features = np.array([])
        total_labels = []
        features = []
        labels = []
        save_path = os.path.join("./cut_features/", "test/", name + "_cut_features.npz")

        # Generate problem instance
        print(f"Generating new {name} instance.\n")
        
        if problem_idx == 0:
            num_elements = 300
            num_sets = 600
            A, b, c, integral_list, sense = generate_setcover(
                num_elements,
                num_sets,
                dens,
                rng,
                max_obj_coeff=max_coef,
                logging=0
            )

        elif problem_idx == 1:
            num_cities = 10
            A, b, c, integral_list, sense = generate_traveling_salesman(
                num_cities,
                rng,
                # max_coord_coeff=max_coef,
                logging=0
            )

        elif problem_idx == 2:
            num_constraints = 15
            num_vars = 20
            A, b, c, integral_list, sense = generate_general_mip(
                num_vars,
                num_constraints,
                dens,
                rng,
                max_constr_coeff=max_coef,
                # max_obj_coeff=max_coef,
                max_solution_value=max_coef,
                logging=0
            )

        elif problem_idx == 3:
            num_items = 700
            A, b, c, integral_list, sense = generate_bounded_knapsack(
                num_items,
                rng,
                max_item_num=max_coef,
                max_v=max_coef,
                max_w=max_coef,
                logging=0
            )

        elif problem_idx == 4:
            num_facilities = 20
            num_demands = 50
            A, b, c, integral_list, sense = generate_facility_location(
                num_facilities,
                num_demands,
                rng,
                # max_demand=max_coef,
                # max_supply=5 * max_coef,
                # max_transport_cost=max_coef,
                # max_facility_cost=max_coef,
                logging=0
            )

        time_list = []
        failed_times = 0
        total_time = 0
        bad_case_flag = False

        print("Testing for no cuts:")
        start_time = time.time()
        print(A.shape)
        obj, sol, time_exceeded = \
            solve(name=name, integral_list=integral_list, sense=sense, A=A, b=b, c=c, verbose=1)
        end_time = time.time()
        time_without_cut = end_time - start_time
        print(f"Time without cut: {time_without_cut:.3f} s")

        print("Check for correctness:")
        sol_mip, obj_mip = \
            solve_mip(name=name, integral_list=integral_list, sense=sense, A=A, b=b, c=c, verbose=0)
        print(f"Norm of solution difference: {np.linalg.norm(sol - sol_mip):.4f}")
        print(f"Objective difference: {obj - obj_mip:.4f}\n")
        
        # Randomly pick cuts and test to solve for `n_step` times
        for j in range(n_steps):
            print("Test idx:", j)

            start_time = time.time()
            succeed_flag, bag_features = solve_with_cut(
                name=name, integral_list=integral_list, sense=sense, A=A, b=b, c=c,
                cut_choice_method=cut_choice_method, cut_percentage=cut_percentage, verbose=1
            )
            end_time = time.time()
            time_with_cut = end_time - start_time

            if not succeed_flag:
                failed_times += 1
                print("Solution fails!\n")
            else:
                time_list.append(time_with_cut)
                total_time += time_with_cut
                features.append(np.mean(bag_features, axis=0))
                print("Training Data Sampled!\n")

            if j >= 5 and failed_times >= 0.8 * j:
                print("Bad case, go to next instance!\n")
                bad_case_flag = True
                break
        
        if not bad_case_flag:
            # Normalize cut features. Need to prevent zero-division in normalization
            features = np.array(features)
            features_mean = np.mean(features, axis=0)
            features_std = np.std(features, axis=0)
            features_std[np.abs(features_std) < 1e-8] = 1
            features_normalized = (features - features_mean) / features_std
        
            mean_time = total_time / (n_steps - failed_times)
            print(f"Mean solving time of {name} instance is {mean_time:.3f} s.")

            labels = np.zeros((features_normalized.shape[0], 2))
            for j in range(features_normalized.shape[0]):
                if time_list[j] < mean_time:
                    labels[j][0] = 1
                else:
                    labels[j][1] = 1

            if total_features.shape[0] == 0:
                total_features = features_normalized
                total_labels = labels
            else:
                total_features = np.concatenate((total_features, features_normalized), axis=0)
                total_labels = np.concatenate((total_labels, labels), axis=0)

            print(total_features)
            print(total_labels)
            np.savez(
                save_path,
                total_features=total_features,
                total_labels=total_labels
            )

    return total_features, total_labels


if __name__ == "__main__":
    if not os.path.exists("./training_bags/test/"):
        os.makedirs("./training_bags/test/")
    main()
