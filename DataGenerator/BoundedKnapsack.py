import random
import numpy as np

def generate_bounded_knapsack(n_items, rng, max_item_num=100, max_v=100, max_w=100, logging=False):
    """Generate Bounded Knapsack problem instance.

    Bounded knapsack problem (with a bounded number of items of the same type).

    :param n_items: number of item types
    :param type: int, should be positive

    :param rng: random number generator
    :param type: `np.random.RandomState` object or other objects that have `randint` method

    :param max_item_num: maximal number for one type of item. By default 100.
    :param type: int, should be positive

    :param max_v: maximal value for one type of item. By default 100.
    :param type: int, should be positive

    :param max_w: maximal weight for one type of item. By default 100.
    :param type: int, should be positive

    :param logging: whether to print the logging info
    :param type: bool
        
    :returns: (A, b, c, integral_list)
        A, b, c: parameter for MIP problem, in standard format `max c @ x,  s.t. A @ x <= b`.
        integral_list: whether the variable is integer. 1 means the variable at the corresponding
            position is integral.
        sense: sense of the objective, "MIN" or "MAX".
    :rtype:
        A: np.array of shape (n + 1, n)
        b: np.array of shape (n,)
        c: np.array of shape (n,)
        integral_list: np.array of shape (n,)
        sense: string
    """
    
    nums = rng.randint(low=1, high=max_item_num, size=(n_items,))
    values = rng.randint(low=1, high=max_v, size=(n_items,))
    weights = rng.randint(low=1, high=max_w, size=(n_items,))
    capacity = np.round(rng.random() * np.sum(weights) * 0.45 + np.sum(weights) * 0.05)

    if logging:
        print("Number of items:\n\t", end="")
        print(nums)
        print("Value of items:\n\t", end="")
        print(values)
        print("Weights of items:\n\t", end="")
        print(weights)
        print("Capacity:\n\t", end="")
        print(capacity)

    # max v @ x,  s.t. x <= num, w @ x <= cap
    A = np.vstack([np.identity(n_items), weights])
    b = np.hstack([nums, capacity])
    c = values

    return A, b, c, np.ones(shape=(n_items,)), "MAX"


def parse_solution_BK(sol, thresh=1e-4):
    print("Selected items:")
    for idx, num in enumerate(sol.astype(int).tolist()):
        if num > thresh:
            print(f"\tItem: {idx}, number: {num}")


if __name__ == "__main__":
    import sys
    import mip
    from mip import Model, CutType, OptimizationStatus

    def create_model(name, A, b, c, integral_array, sense):
        # Params
        num_constrs, num_vars = A.shape
        # Create a new model
        m = Model(sense=sense, solver_name=mip.CBC)
        # Create variables
        x = []
        for i, x_type in enumerate(integral_array.tolist()):
            x_type = mip.INTEGER if x_type > 0 else mip.CONTINUOUS
            x.append(m.add_var(var_type=x_type, name=f"x_{i}"))
        # Set objective
        m.objective = mip.xsum(c[i] * x[i] for i in range(num_vars))
        # Add constraints
        constr = [m.add_constr(mip.xsum(A[i, j] * x[j] for j in range(num_vars)) <= b[i],
                            name=f"C_{i}") for i in range(num_constrs)]
        return m, x, constr

    A, b, c, integral_list, sense = \
        generate_bounded_knapsack(20, np.random.RandomState(0), logging=True)

    m, x, constr = create_model(
        name="TestBoundedKnapsack",
        A=A,
        b=b,
        c=c,
        integral_array=integral_list,
        sense=sense,
    )

    print()
    status = m.optimize(max_seconds=120)
    if status == mip.OptimizationStatus.OPTIMAL:
        print(f"Optimal solution cost {m.objective_value} found")
    elif status == mip.OptimizationStatus.FEASIBLE:
        print(f"Sol.cost {m.objective_value} found, best possible: {m.objective_bound}")
    elif status == mip.OptimizationStatus.NO_SOLUTION_FOUND:
        print(f"No feasible solution found, lower bound is: {m.objective_bound}")
    if status == mip.OptimizationStatus.OPTIMAL or status == mip.OptimizationStatus.FEASIBLE:
        sol = np.array([v.x for v in m.vars])
        print("Solution:\n\t", end="")
        parse_solution_BK(sol)
