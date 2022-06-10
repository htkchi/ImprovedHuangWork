import random
import numpy as np


def generate_setcover(num_elements, num_sets, dens, rng, max_obj_coeff=100, logging=False):
    """Generate set cover problem instance.

    TODO: fill in generation details.

    :param num_elements: number of variables
    :param type: int, should be positive

    :param num_sets: number of constraints
    :param type: int, should be positive

    :param dens: density of constraint matrix
    :param type: float, should be in (0, 1)

    :param rng: random number generator
    :param type: `np.random.RandomState` object or other objects that have `randint` method

    :param max_obj_coeff: maximal value of objective coefficient. By default 100.
    :param type: int, should be positive

    :param logging: whether to print the logging info
    :param type: bool
        
    :returns: (A, b, c, integral_list, sense)
        A, b, c: parameter for MIP problem, in standard format `max c @ x,  s.t. A @ x <= b`.
        integral_list: whether the variable is integer. 1 means the variable at the corresponding
            position is integral.
        sense: sense of the objective, "MIN" or "MAX".
    :rtype:
        A: np.array of shape (num_elements, num_sets)
        b: np.array of shape (num_sets,)
        c: np.array of shape (num_sets,)
        integral_list: np.array of shape (num_sets,)
        sense: string
    """
    # Need at least 1 non-zero entry per row, and at least 2 non-zero entries per column.
    # Otherwise the constraint / variable is inactive, and can be removed.
    # So the density should at least satisfies the below inequality.
    nnzrs = int(num_elements * num_sets * dens)
    assert nnzrs >= num_elements and nnzrs >= 2 * num_sets

    # Find the number of non-zero entries for each row
    row_nz_elem_split = sorted(rng.choice(np.arange(1, nnzrs+1), num_elements, replace=False),
                               reverse=True) + [0]
    row_nz_elem_num = np.array([row_nz_elem_split[i] - row_nz_elem_split[i+1] 
                                for i in range(num_elements)])
    
    # Find the column index of non-zero entries for each row
    col_idx_list = rng.randint(low=0, high=num_sets, size=(nnzrs,))
    # Ensure at least 2 non-zero entries per column
    col_idx_list[rng.choice(nnzrs, num_sets * 2, replace=False)] = \
        np.repeat(np.arange(num_sets), 2)

    if logging:
        print("Number of non-zero elements for each row:\n\t", end="")
        print(row_nz_elem_num)
        print("Column index of non-zero elements:\n\t", end="")
        print(col_idx_list)
    
    A = np.zeros(shape=(num_elements, num_sets))
    elem_idx = 0
    for i in range(num_elements):
        A[i, col_idx_list[elem_idx : elem_idx + row_nz_elem_num[i]]] = -1
        elem_idx += row_nz_elem_num[i]
    b = -np.ones(shape=(num_elements,))
    c = rng.randint(low=1, high=max_obj_coeff, size=num_sets)

    if logging:
        print("A:\n\t", A)
        print("b:\n\t", b)
        print("c:\n\t", c)

    return A, b, c, np.ones(shape=(num_sets,)), "MIN"


def parse_solution_SC(sol, thresh=1e-4, **kwargs):
    print("Selected sets: " + ", ".join(map(str, np.argwhere(sol > thresh).astype(int))))


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
        generate_setcover(5, 10, 0.4, np.random.RandomState(0), logging=True)

    m, x, constr = create_model(
        name="TestSetCover",
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
        parse_solution_SC(sol)
