from gauss_elimination import back_substitution, forward_substitution
import copy
import numpy as np
from Model.MatrixSolver import MatrixSolver

def LU_decomposition(A, b=[], scaling=False):
    """
    Perform LU decomposition of A with optional scaled partial pivoting.
    A : square matrix (list of lists)
    scaling : if True → use scaled partial pivoting, 
              if False → use normal partial pivoting
    """
    n = len(A)
    A = copy.deepcopy(A)
    L = [[0.0]*n for _ in range(n)]
    U = [[0.0]*n for _ in range(n)]
    P = list(range(n))  # permutation vector
    steps = []

    steps.append("Initialized P = " + str(P))

    # Scaling factors (only used when scaling=True)
    if scaling:
        s = [max(abs(A[i][j]) for j in range(n)) for i in range(n)]
    else:
        # No scaling → set all scaling factors = 1
        s = [1 for _ in range(n)]

    er = 1 # -1 if no solution, 0 if infinite
    rankA = np.linalg.matrix_rank(A)
    
    b_ = np.array(b).reshape(len(b), 1) 
    augumented = np.hstack((A, b_))
    rankAb = np.linalg.matrix_rank(augumented)
    
    if rankA != rankAb: #no solution
        er = -1
    elif rankA == rankAb and rankA < n: #inf solutions
        er = 0    

    for k in range(n-1):

        # --- PIVOT SELECTION ---
        pivot_row = k
        max_ratio = 0
        # ratio = |A[i][k]| / s[i]  (if scaling) OR |A[i][k]| (if no scaling)
        if scaling:
            if s[P[k]] != 0:
                max_ratio = abs(A[P[k]][k]) / s[P[k]]
        else:
            max_ratio = abs(A[P[k]][k])

        for i in range(k+1, n):
            ratio = 0
            if scaling:
                if s[P[i]] != 0:
                    ratio = abs(A[P[i]][k]) / s[P[i]]    
            else:
                ratio = abs(A[P[i]][k])

            if ratio > max_ratio:
                max_ratio = ratio
                pivot_row = i
        

        # Swap rows in permutation vector
        P[k], P[pivot_row] = P[pivot_row], P[k]
        steps.append(f"Pivoting: Swapped row {k} with row {pivot_row}, (New P: {P})")

        # --- ELIMINATION ---
        for i in range(k+1, n):
            if A[P[k]][k] != 0:
                factor = A[P[i]][k] / A[P[k]][k]
                A[P[i]][k] = factor  # store factor (L)

                steps.append({
                    "type": "calc_1",
                    "i":"i",
                    "j":k,
                    "formula":f"L_{{{i}{k}}} = A_{{{i}{k}}} / A_{{{k}{k}}}",
                    "res": MatrixSolver._fmt(factor)
                })

                for j in range(k+1, n):
                    A[P[i]][j] -= factor * A[P[k]][j]

    # Build L and U
    for i in range(n):
        for j in range(n):
            if j < i:
                L[i][j] = A[P[i]][j]
            elif j == i:
                L[i][j] = 1.0
                U[i][j] = A[P[i]][j]
            else:
                U[i][j] = A[P[i]][j]

    return L, U, P, er, steps



def solve_from_LU(A, b, scaling):
    L, U, P, er, steps = LU_decomposition(A, b, scaling)
    if er == -1:
        return "No solution"  
    elif er == 0:
        return "Infinite solutions"
    
    n = len(A)
    # Apply row permutation to b
    b_permuted = [b[P[i]] for i in range(n)]    # Solve Ly = Pb
    y = forward_substitution(L, b_permuted, n)
    # Solve Ux = y
    x = back_substitution(U, y, n)
    return x



