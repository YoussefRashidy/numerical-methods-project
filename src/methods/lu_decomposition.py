from gauss_elimination import back_substitution, forward_substitution
import copy

def LU_decomposition(A, tol=1e-12, scaling=False):
    """
    Perform LU decomposition of A with optional scaled partial pivoting.
    A : square matrix (list of lists)
    tol : tolerance for singularity
    scaling : if True → use scaled partial pivoting, 
              if False → use normal partial pivoting
    """
    n = len(A)
    A = copy.deepcopy(A)
    L = [[0.0]*n for _ in range(n)]
    U = [[0.0]*n for _ in range(n)]
    P = list(range(n))  # permutation vector

    # Scaling factors (only used when scaling=True)
    if scaling:
        s = [max(abs(A[i][j]) for j in range(n)) for i in range(n)]
    else:
        # No scaling → set all scaling factors = 1
        s = [1 for _ in range(n)]

    er = 0

    for k in range(n-1):

        # --- PIVOT SELECTION ---
        pivot_row = k

        # ratio = |A[i][k]| / s[i]  (if scaling) OR |A[i][k]| (if no scaling)
        if scaling:
            max_ratio = abs(A[P[k]][k]) / s[P[k]]
        else:
            max_ratio = abs(A[P[k]][k])

        for i in range(k+1, n):
            if scaling:
                ratio = abs(A[P[i]][k]) / s[P[i]]
            else:
                ratio = abs(A[P[i]][k])

            if ratio > max_ratio:
                max_ratio = ratio
                pivot_row = i

        if max_ratio < tol:
            er = -1
            return None, None, None, er

        # Swap rows in permutation vector
        P[k], P[pivot_row] = P[pivot_row], P[k]

        # --- ELIMINATION ---
        for i in range(k+1, n):
            factor = A[P[i]][k] / A[P[k]][k]
            A[P[i]][k] = factor  # store factor (L)
            for j in range(k+1, n):
                A[P[i]][j] -= factor * A[P[k]][j]

    if abs(A[P[n-1]][n-1]) < tol:
        er = -1
        return None, None, None, er

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

    return L, U, P, er



def solve_from_LU(A, b):
    L, U, P, er = LU_decomposition(A)
    if er == -1:
        return None  # Singular matrix
    n = len(A)
    # Apply row permutation to b
    b_permuted = [b[P[i]] for i in range(n)]    # Solve Ly = Pb
    y = forward_substitution(L, b_permuted, n)
    # Solve Ux = y
    x = back_substitution(U, y, n)
    return x



def Crout(A, tol=1e-12, scaling=False):
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

    er = 0

    for k in range(n):

        # ---- Pivoting ----
        pivot_row = k
        max_ratio = abs(A[P[k]][k]) / s[P[k]]

        for i in range(k+1, n):
            ratio = abs(A[P[i]][k]) / s[P[i]]
            if ratio > max_ratio:
                max_ratio = ratio
                pivot_row = i

        if max_ratio < tol:
            return None, None, None, -1  # Singular

        # Swap permutation
        P[k], P[pivot_row] = P[pivot_row], P[k]

        # ---- Compute L column k ----
        L[P[k]][k] = A[P[k]][k]
        U[P[k]][k] = 1.0     # diagonal of U = 1

        for i in range(k+1, n):
            L[P[i]][k] = A[P[i]][k]

        # ---- Compute U row k ----
        for j in range(k+1, n):
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



def solve_from_Crout(A, b):
    L, U, P, er = Crout(A)
    if er == -1:
        return None
    n = len(A)
    # Permute b
    b_permuted = [b[P[i]] for i in range(n)]
    y = forward_substitution(L, b_permuted, n)
    x = back_substitution(U, y, n)
    return x
