import random
import numpy as np


def generate_traveling_salesman(num_cities, rng, max_coord_coeff=100, logging=False):
    """Generate travling salesman problem instance.

    Using Miller–Tucker–Zemlin formulation.
        Tucker, A. W. (1960), "On Directed Graphs and Integer Programs", IBM Mathematical research
        Project (Princeton University).

    :param num_cities: number of cities
    :param type: int, should be positive

    :param rng: random number generator
    :param type: `np.random.RandomState` object or other objects that have `randint` method

    :param max_coord_coeff: maximal value for coordinate coefficient for city positions
    :param type: int, should be positive

    :param logging: whether to print the logging info
    :param type: bool
        
    :returns: (A, b, c, dist_matrix, integral_list)
        A, b, c: parameter for MIP problem, in standard format `min c @ x,  s.t. A @ x <= b`.
        integral_list: whether the variable is integer. 1 means the variable at the corresponding
            position is integral.
        sense: sense of the objective, "MIN" or "MAX".
    :rtype:
        A: np.array of shape (2 * n * n + 4, n * n + n)
        b: np.array of shape (2 * n * n + 4,)
        c: np.array of shape (n * n + n,)
        integral_list: np.array of shape (n * n + n,)
        sense: string
    """

    # Generate cities distance matrix
    city_pos = rng.randint(size=(num_cities, 2), low=0, high=max_coord_coeff)
    dist_matrix = np.array(
        [[np.linalg.norm(city_pos[i] - city_pos[j]) for j in range(num_cities)]
         for i in range(num_cities)],
        dtype=np.float32
    )
    if logging:
        print(f"In total {num_cities} generated with position")
        print(city_pos)
        print("Distance matrix:")
        print(dist_matrix)

    # Variable: X = [x_{0,0}, x_{0,1}, ..., x_{0,n-1}, ..., x_{n-1,n-1}, u_{0}, ..., u_{n-1}]
    # x_{i,j} \in {0, 1}; u_{i} \in {1, ..., n-1}
    
    # Objective: minimize sum(dist_{i,j} * x_{i,j})
    c = dist_matrix.flatten()
    c = np.hstack([c, np.zeros(shape=(num_cities,))])

    # Constraint 1: FORALL i: x_{i,0} + ... + x_{i,i-1} + x_{i,i+1} + ... + x_{i,n-1} == 1
    A_1 = np.zeros(shape=(num_cities, num_cities * num_cities + num_cities))
    for i in range(num_cities):
        A_1[i, i*num_cities:(i+1)*num_cities] = 1  
        A_1[i, i*num_cities+i] = 0
    A_1 = np.vstack([A_1, -A_1])
    b_1 = np.hstack([np.ones(shape=(num_cities,)) + 1e-6, -np.ones(shape=(num_cities,)) - 1e-6])

    # Constraint 2: FORALL j: x_{0,j} + ... + x_{j-1,j} + x_{j+1,j} + ... + x_{n-1,j} == 1
    A_2 = np.zeros(shape=(num_cities, num_cities * num_cities + num_cities))
    for j in range(num_cities):
        A_2[j, j:num_cities*num_cities+j:num_cities] = 1  
        A_2[j, j*num_cities+j] = 0
    A_2 = np.vstack([A_2, -A_2])
    b_2 = np.hstack([np.ones(shape=(num_cities,)) + 1e-6, -np.ones(shape=(num_cities,)) - 1e-6])
    
    # Constraint 3 (MTZ): FORALL 1 <= i != j < num_cities: u_{i} - u_{j} + n*x_{i,j} <= n - 1
    # Default start point is x_{0}
    A_3 = np.zeros(shape=(num_cities * num_cities, num_cities * num_cities + num_cities))
    for i in range(1, num_cities):
        for j in range(1, i):
            A_3[i*num_cities + j, i*num_cities + j] = num_cities
            A_3[i*num_cities + j, num_cities*num_cities + i] = 1
            A_3[i*num_cities + j, num_cities*num_cities + j] = -1
        for j in range(i+1, num_cities):
            A_3[i*num_cities + j, i*num_cities + j] = num_cities
            A_3[i*num_cities + j, num_cities*num_cities + i] = 1
            A_3[i*num_cities + j, num_cities*num_cities + j] = -1
    b_3 = np.ones(shape=(num_cities * num_cities,)) * (num_cities - 1)

    # Constraint 4 (MTZ Dummy Var): FORALL 1 <= i < num_cities: 1 <= u_{i} <= n-1
    # u_{i} denotes the visited sequence of cities. u_{i} == t means city i is visited at time t
    A_4 = np.zeros(shape=(num_cities * 2, num_cities * num_cities + num_cities))
    for i in range(1, num_cities):
        A_4[i, num_cities*num_cities + i] = 1
        A_4[i + num_cities, num_cities*num_cities + i] = -1
    b_4 = np.hstack([np.ones(shape=(num_cities,)) * (num_cities - 1), -np.ones(shape=(num_cities,))])

    # Constraint 5 (Binary): FORALL i, j: 0 <= x_{i,j} <= 1
    A_5 = np.zeros(shape=(num_cities * num_cities, num_cities * num_cities + num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            A_5[i*num_cities + j, i*num_cities + j] = 1
    b_5 = np.ones(shape=(num_cities * num_cities,))

    A = np.vstack([A_1, A_2, A_3, A_4, A_5])
    b = np.hstack([b_1, b_2, b_3, b_4, b_5])
    
    # return A, b, c, dist_matrix, np.ones(shape=(A.shape[1]))
    return A, b, c, np.ones(shape=(A.shape[1])), "MIN"


def parse_solution_TS(sol, obj=None, c=None, thresh=0.01, **kwargs):
    num_cities = int(np.sqrt(num_cities))
    nonzero_x = np.argwhere(sol[:num_cities*num_cities] > 1e-4).astype(int)
    nonzero_pos = np.hstack([nonzero_x // num_cities, nonzero_x % num_cities])
    u = sol[num_cities*num_cities+1:]
    seq = np.argsort(u) + 1
    if obj is not None and c is not None:
        idx = [0+seq[0]] + \
              [seq[i]*num_cities + seq[i+1] for i in range(len(seq)-1)] + \
              [seq[-1]*num_cities]
        assert sum([c[i] for i in idx]) - obj <= thresh
    seq = ["0"] + list(map(str, seq.astype(int).tolist()))
    print("Visiting sequence:")
    print("\t" + ", ".join(seq))
    print("Non zero entries:")
    print("\t" + str(nonzero_pos.tolist()))
    return seq


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
        generate_traveling_salesman(10, np.random.RandomState(0), logging=False)

    m, x, constr = create_model(
        name="TestTSP",
        A=A,
        b=b,
        c=c,
        integral_array=integral_list,
        sense=sense
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
        obj = m.objective_value
        parse_solution_TS(sol, 10, obj, c)
