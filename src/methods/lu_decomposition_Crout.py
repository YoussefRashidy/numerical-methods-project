from src.methods.gauss_elimination import back_substitution, forward_substitution
import copy
import numpy as np
from Model.MatrixSolver import MatrixSolver

def Crout(A, b, sig_figs,scaling=False):
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
    steps = []

    steps.append(f"<div class='text-slate-400'>Initialized P = {P}</div>")

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
        steps.append(
                    f"<div class='p-2 my-2 bg-slate-800/50 border-l-2 border-yellow-500 rounded'>"
                    f"<span class='text-yellow-400 font-bold'>Pivoting:</span> Swapped Row ${k}$ $\\leftrightarrow$ Row ${pivot_row}$ "
                    f"<br><span class='text-xs text-slate-500'>New Permutation $P$: {P}</span>"
                    f"</div>"
                )

        # ---- Compute L column k ----
        L[P[k]][k] = A[P[k]][k]
        U[P[k]][k] = 1.0     # diagonal of U = 1

        steps.append({
                "type": "calc_l", "i": k, "j": k,
                "formula": f"L_{{{k}{k}}} = A_{{{k}{k}}}",
                "res": MatrixSolver._fmt(L[P[k]][k], sig_figs)
            })

        for i in range(k+1, n):
            L[P[i]][k] = A[P[i]][k]
            steps.append({
                    "type": "calc_l", "i": i, "j": k,
                    "formula": f"L_{{{i}{k}}} = A_{{{i}{k}}}",
                    "res": MatrixSolver._fmt(L[P[i]][k], sig_figs)
                })

        # ---- Compute U row k ----
        for j in range(k+1, n):
            if L[P[k]][k] != 0:
                U[P[k]][j] = A[P[k]][j] / L[P[k]][k]
                steps.append({
                        "type": "calc_u", "i": k, "j": j,
                        "formula": f"U_{{{k}{j}}} = A_{{{k}{j}}} / L_{{{k}{k}}}",
                        "res": MatrixSolver._fmt(U[P[k]][j], sig_figs)
                    })
            else:
                steps.append(f"<div class='text-red-400'>Warning: Zero pivot encountered at L_{{{k}{k}}}</div>")

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

    return L_final, U_final, P, er, steps



def solve_from_Crout(A, b, scaling, sig_figs=4):
    L, U, P, er, steps = Crout(A, b, sig_figs, scaling)
    if er == -1:
        return "No solution"  
    elif er == 0:
        return "Infinite solutions"
    
    n = len(A)
    # Permute b
    b_permuted = [b[P[i]] for i in range(n)]
    steps.append(f"<div class='mt-4 pt-4 border-t border-slate-700 font-bold text-cyan-400'>Solving Phase</div>")
    steps.append(f"Permuted vector $b$ according to $P$: ${MatrixSolver._fmt_vec(b_permuted, sig_figs)}$")
    
    y = forward_substitution(L, b_permuted, n)
    steps.append(f"Forward Substitution Result ($y$): ${MatrixSolver._fmt_vec(y, sig_figs)}$")
    x = back_substitution(U, y, n)
    return x, L, U, steps


