import random
import numpy as np


def generate_facility_location(num_facilities, num_demands, rng, max_demand=100, max_supply=500,
                               max_transport_cost=100, max_facility_cost=100, logging=False):
    """Generate capacitated facility location problem instance.

    Following https://en.wikipedia.org/wiki/Facility_location_problem formulation.

    TODO: fill in generation details.

    :param num_factories: number of variables
    :param type: int, should be positive

    :param num_demands: number of constraints
    :param type: int, should be positive

    :param rng: random number generator
    :param type: `np.random.RandomState` object or other objects that have `randint` method

    :param max_demand: maximal value of demand of each kind. By default 100.
    :param type: int, should be positive

    :param max_supply: maximal value of supply by a single facility. By default 500.
    :param type: int, should be positive

    :param max_transport_cost: maximal coefficient of transportation cost. By default 5.
                               This total cost of transportation will be 
                               `sum_{i, j} [c_{ij} d_{j} y_{ij}]`, where this parameter corresponds
                               to the maximal value of `c_{ij}`.
    :param type: int, should be positive

    :param max_facility_cost: maximal coefficient of facility opening cost. By default 5.
                              This total cost of facility opening will be `sum_{i} [f_{i} x_{i}]`，
                              where this param corresponds to the maximal value of `c_{ij}`.
    :param type: int, should be positive

    :param logging: whether to print the logging info
    :param type: bool

    :returns: (A, b, c, integral_list, sense)
        A, b, c: parameter for MIP problem, in standard format `min c @ x,  s.t. A @ x <= b`.
        integral_list: whether the variable is integer. 1 means the variable at the corresponding
            position is integral.
        sense: sense of the objective, "MIN" or "MAX".
    :rtype:
        A: np.array of shape (num_constrs, num_vars)
        b: np.array of shape (num_vars,)
        c: np.array of shape (num_vars,)
        integral_list: np.array of shape (num_vars,)
        sense: string
    """

    demand = (0.9 * rng.rand(num_demands) + 0.1) * max_demand
    supply = (0.9 * rng.rand(num_facilities) + 0.1) * max_supply

    # N facilities and M demands
    # Variable: X = [x_{0}, ..., x_{N-1}, y_{0,0}, y_{0,1}, ..., y_{0,M-1}, ..., y_{N-1,M-1}]
    # x_{i} \in {0, 1}; y_{i, j} >= 0
    integral_array = np.hstack([np.ones(num_facilities), np.zeros(num_facilities * num_demands)])
    
    # Objective: minimize sum_{i, j}(c_{i,j} * d_{i} *  y_{i,j}) + sum_{i}(f_{i} * x_{i})
    transport_cost = rng.rand(num_facilities, num_demands) * max_transport_cost
    facility_cost = rng.rand(num_facilities) * max_facility_cost
    c = np.hstack([facility_cost, transport_cost.flatten()])

    # Constraint 1: FORALL j: sum_{i} (y_{i,j}) == 1
    A_1 = np.zeros(shape=(num_demands, num_facilities * (num_demands + 1)))
    for j in range(num_demands):
        A_1[j, num_facilities + j :: num_demands] = 1
    A_1 = np.vstack([A_1, -A_1])
    b_1 = np.hstack([np.ones(shape=(num_demands,)), -np.ones(shape=(num_demands,))])

    # Constraint 2: FORALL i: sum_{j} (d_{j} y_{i, j}) <= u_{i} x_{i}
    A_2 = np.zeros(shape=(num_facilities, num_facilities * (num_demands + 1)))
    for i in range(num_facilities):
        A_2[i, i] = -supply[i]
        A_2[i, num_facilities + num_demands * i : num_facilities + num_demands * (i + 1)] = demand
    b_2 = np.zeros(shape=(num_facilities,))

    # Constraint 3: FORALL i: 0 <= x_{i} <= 1
    A_3 = np.zeros(shape=(num_facilities, num_facilities * (num_demands + 1)))
    for i in range(num_facilities):
        A_3[i, i] = 1
    b_3 = np.ones(shape=(num_facilities,))

    A = np.vstack([A_1, A_2, A_3])
    b = np.hstack([b_1, b_2, b_3])

    if logging:
        print("A:\n", A)
        print("b:\n\t", b)
        print("c:\n\t", c)
        print("integral_array:\n\t", integral_array)

    return A, b, c, integral_array, "MIN"


def parse_solution_FL(sol, thresh=1e-4, **kwargs):
    nonzero_x = np.argwhere(sol[:n_facilities] > 1 - thresh).flatten().astype(int)
    nonzero_y = np.round(
        sol[n_facilities:], int(-np.log(thresh) + 1)
    ).reshape((n_facilities, n_demands))
    print("Opened facilities:")
    print("\t" + ", ".join(map(str, nonzero_x.tolist())))
    print("Transportation ratio:")
    for j in range(n_demands):
        print(f"\tDemand {j}:\t" + ", ".join([f"facility {i}: {ratio:.2%}" 
            for i, ratio in enumerate(nonzero_y[:, j].tolist()) if ratio > thresh]))


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

    A, b, c, integral_array, sense = \
        generate_facility_location(5, 10, np.random.RandomState(0), logging=True)

    m, x, constr = create_model(
        name="TestFacilityLocation",
        A=A,
        b=b,
        c=c,
        integral_array=integral_array,
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
        parse_solution_FL(sol, 5, 10)
