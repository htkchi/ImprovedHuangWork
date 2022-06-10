import random
import numpy as np


def generate_general_mip(num_vars, num_constrs, dens, rng, max_constr_coeff=100, max_obj_coeff=100,
                         max_solution_value=100, ensure_solution=True, logging=False):
    """Generate general MIP instance.

    TODO: fill in generation details.

    :param num_vars: number of variables
    :param type: int, should be positive

    :param num_constrs: number of constraints
    :param type: int, should be positive

    :param dens: density of constraint matrix
    :param type: float, should be in (0, 1)

    :param rng: random number generator
    :param type: `np.random.RandomState` object or other objects that have `randint` method

    :param max_constr_coeff: maximal value of constraints coefficient. By default 100.
    :param type: int, should be positive

    :param max_obj_coeff: maximal value of objective coefficient. By default 100.
    :param type: int, should be positive

    :param max_solution_value: maximal value of solution coefficient. By default 100.
    :param type: int, should be positive

    :param ensure_solution: whether to ensure a integral solution in the problem. By default True
    :param type: bool

    :param logging: whether to print the logging info
    :param type: bool
        
    :returns: (A, b, c, integral_list)
        A, b, c: parameter for MIP problem, in standard format `max c @ x,  s.t. A @ x <= b`.
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
    nnzrs = int(num_constrs * num_vars * dens)
    assert nnzrs >= num_constrs and nnzrs >= 2 * num_vars

    row_nz_elem_split = sorted(rng.choice(np.arange(1, nnzrs), num_constrs, replace=False).tolist(), reverse=True) + [0]
    row_nz_elem_num = np.array([row_nz_elem_split[i] - row_nz_elem_split[i+1] for i in range(num_constrs)])
    col_idx_list = rng.randint(low=0, high=num_vars, size=(nnzrs,))
    col_idx_list[rng.choice(nnzrs, num_vars * 2, replace=False)] = np.repeat(np.arange(num_vars), 2)

    if logging:
        print("Number of non-zero elements for each row:\n\t", end="")
        print(row_nz_elem_num)
        print("Column index of non-zero elements:\n\t", end="")
        print(col_idx_list)
    
    if ensure_solution:
        ensured_solution = rng.randint(low=0, high=max_solution_value, size=(num_vars,))
        if logging:
            print("Ensured solution:\n\t", ensured_solution)

    A_list = []
    b_list = []
    elem_idx = 0
    for i in range(num_constrs):
        a = np.zeros(shape=(num_vars,))
        a[col_idx_list[elem_idx : elem_idx + row_nz_elem_num[i]]] = \
            rng.randint(low=1, high=max_constr_coeff, size=(row_nz_elem_num[i],))
        if ensure_solution:
            b = rng.randint(low=a @ ensured_solution.T, high=5 * a @ ensured_solution.T)
        else:
            b = rng.randint(high=max_constr_coeff * max_solution_value)
        A_list.append(a)
        b_list.append(b)
        elem_idx += row_nz_elem_num[i]

    A = np.vstack(A_list)
    b = np.hstack(b_list)
    c = rng.randint(low=1, high=max_obj_coeff, size=num_vars)

    if logging:
        print("A:\n\t", A)
        print("b:\n\t", b)
        print("c:\n\t", c)
        if ensure_solution:
            print("Checking A @ ensured_solution <= b:\n\t", np.all(A @ ensured_solution.T <= b))

    return A, b, c, np.ones(shape=(num_vars,)), "MAX"


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
        generate_general_mip(10, 20, 0.2, np.random.RandomState(0), logging=True)

    m, x, constr = create_model(
        name="TestGeneralMIP",
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
        print(sol)
