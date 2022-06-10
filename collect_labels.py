import os
import sys
import time
import numpy as np

from branch_and_bound import solve_with_cut

from DataGenerator.BoundedKnapsack import generate_bounded_knapsack
from DataGenerator.FacilityLocation import generate_facility_location
from DataGenerator.GeneralMIP import generate_general_mip
from DataGenerator.SetCover import generate_setcover
from DataGenerator.TravelingSalesman import generate_traveling_salesman

def collect_labels(n_instances, n_steps, problem_idx, **kwargs):
    problem_names = [
        "set_cover",
        "traveling_salesman",
        "general_mip",
        "knapsack",
        "facility_location"
    ]
    name = problem_names[problem_idx]
    n_features = 14
    dens = 0.3
    max_coef = 10
    total_features = np.array([])
    total_labels = []
    #cut_choice_method = "CollectLabels"
    cut_choice_method = kwargs.get("cut_choice_method", "CollectLabels")
    cut_percentage = 0.3
    save_path = kwargs.get("save_path", f"./training_bags/{name}_training_bags.npz")
    t = time.strftime("%Y%m%d_%H%M%S", time.gmtime())

    for i in range(0, n_instances):
        rng = np.random.RandomState(i)
        features = []
        labels = []
        total_time = 0

        # Generate problem instance
        print(f"Generating new {name} instance.")
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
            num_vars = 30
            A, b, c, integral_list, sense = generate_general_mip(
                num_vars,
                num_constraints,
                dens,
                rng,
                # max_constr_coeff=max_coef,
                max_obj_coeff=max_coef,
                max_solution_value=max_coef,
                logging=0
            )
        elif problem_idx == 3:
            num_items = 700
            A, b, c, integral_list, sense = generate_bounded_knapsack(
                num_items,
                rng,
                # max_item_num=max_coef,
                # max_v=max_coef,
                # max_w=max_coef,
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
        bad_case_flag = False

        # Randomly pick cuts and test to solve for some times
        for j in range(n_steps):
            print("Instance idx:", i, ", Test idx:", j)
            start_time = time.time()
            succeed_flag, bag_features = solve_with_cut(
                name=name, integral_list=integral_list, sense=sense, A=A, b=b, c=c,
                cut_choice_method=cut_choice_method, cut_percentage=cut_percentage,
                solution_parser=print, verbose=1
            )
            end_time = time.time()
            time_with_cut = end_time - start_time

            if not succeed_flag:
                failed_times += 1
                print("Solution fails!\n")
            else:
                total_time += time_with_cut
                time_list.append(time_with_cut)
                features.append(np.mean(bag_features, axis=0))
                print("Training data sampled!\n")
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
            print(f"Mean solving time of {name}, {i} instance is {mean_time:.4f} s.")

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

            print(f"Append {features_normalized.shape[0]} rows in feature array.\n\n")

        np.savez(
            os.path.join(save_path, f"{name}_training_bags_{i}of{n_instances}_{t}.npz"),
            features=total_features,
            labels=total_labels
        )
    # with open(os.path.join(save_path, f"{name}_logs_{t}.txt"), "r") as f:
    #     f.write(f"Logs for {name} problem instance at {t}\n")

    return total_features, total_labels, \
           os.path.join(save_path, f"{name}_training_bags_{n_instances-1}of{n_instances}_{t}.npz")


def main():
    problem_names = [
        "set_cover",
        "traveling_salesman",  # This has some bug, ignore it.
        "general_mip",
        "knapsack",
        "facility_location"
    ]
    problem_idx = 0
    # save_path = os.path.join(f"./training_bags/{problem_names[problem_idx]}/")
    save_path = os.path.join(f"./training_bags/")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    n_instances = 100
    n_steps = 100
    cut_choice_method = "CollectLabels"   # Collect Training Data
    #cut_choice_method = "Test"    # Evaluate
    _, _, file_path = collect_labels(n_instances, n_steps, problem_idx, cut_choice_method=cut_choice_method, save_path=save_path)

    training_data = np.load(file_path)
    features = training_data["features"]
    labels = training_data["labels"]

    print(features.shape)
    print(labels.shape)


if __name__ == "__main__":
    main()
