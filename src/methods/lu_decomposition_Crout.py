from gauss_elimination import back_substitution, forward_substitution
import copy
import numpy as np

def Crout(A, b, scaling=False):
    """
    Perform Crout LU decomposition of A with optional scaled partial pivoting.
    L : lower-triangular matrix (Crout: diagonal NOT 1)
    U : upper-triangular matrix with diagonal = 1
    P : permutation vector
    er : -1 if singular, 0 otherwise
    """
    n = len(A)
    A = copy.deepcopy(A)
    L = [[0.0]*n for _ in range(n)]
    U = [[0.0]*n for _ in range(n)]
    P = list(range(n))

    # Scaling factors (only used if scaling=True)
    if scaling:
        s = [max(abs(A[i][j]) for j in range(n)) for i in range(n)]
    else:
        s = [1]*n   # neutral scaling

    er = 1 # -1 if no solution, 0 if infinite
    rankA = np.linalg.matrix_rank(A)
    
    b_ = np.array(b).reshape(len(b), 1) 
    augumented = np.hstack((A, b_))
    rankAb = np.linalg.matrix_rank(augumented)
    
    if rankA != rankAb: #no solution
        er = -1
    elif rankA == rankAb and rankA < n: #inf solutions
        er = 0    
        
    for k in range(n):

        # ---- Pivoting ----
        pivot_row = k
        max_ratio = 0
        if s[P[k]] != 0:
            max_ratio = abs(A[P[k]][k]) / s[P[k]]

        for i in range(k+1, n):
            if s[P[i]] != 0:
                ratio = abs(A[P[i]][k]) / s[P[i]]
                if ratio > max_ratio:
                    max_ratio = ratio
                    pivot_row = i
        

        # Swap permutation
        P[k], P[pivot_row] = P[pivot_row], P[k]

        # ---- Compute L column k ----
        L[P[k]][k] = A[P[k]][k]
        U[P[k]][k] = 1.0     # diagonal of U = 1

        for i in range(k+1, n):
            L[P[i]][k] = A[P[i]][k]

        # ---- Compute U row k ----
        for j in range(k+1, n):
            if L[P[k]][k] != 0:
                U[P[k]][j] = A[P[k]][j] / L[P[k]][k]

        # ---- Update submatrix ----
        for i in range(k+1, n):
            for j in range(k+1, n):
                A[P[i]][j] -= L[P[i]][k] * U[P[k]][j]

    # Rebuild L and U in correct order
    L_final = [[0.0]*n for _ in range(n)]
    U_final = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if j <= i:
                L_final[i][j] = L[P[i]][j]
            if j >= i:
                U_final[i][j] = U[P[i]][j]

    return L_final, U_final, P, er



def solve_from_Crout(A, b, scaling):
    L, U, P, er = Crout(A, b, scaling)
    if er == -1:
        return "No solution"  
    elif er == 0:
        return "Infinite solutions"
    
    n = len(A)
    # Permute b
    b_permuted = [b[P[i]] for i in range(n)]
    y = forward_substitution(L, b_permuted, n)
    x = back_substitution(U, y, n)
    return x


