import math

from src.utils.precision_rounding import toDecimal, toFloats, intializeContext
import decimal
from src.methods.gauss_elimination import back_substitution,forward_substitution
import copy
import numpy as np


class MatrixSolver:

    @staticmethod
    def isDiagonalyDominant(A):
        #Check if Matrix A is diagonaly dominant 
        num_rows = len(A)
        num_columns = len(A[0])
        # Flags to check system
        at_least_one_strictly_greater = None 
        if(num_rows != num_columns) : raise Exception("Matrix must be square")
        for i in range(num_rows) :
            nonDiagonalSum = 0 
            for j in range(num_rows) :
                if(j==i) : continue 
                nonDiagonalSum += abs(A[i][j]) 
            if(A[i][i] > nonDiagonalSum) :
                at_least_one_strictly_greater = True
            elif(A[i][i] == nonDiagonalSum) :
                if(at_least_one_strictly_greater == None) : at_least_one_strictly_greater = False
            else :
                return False    
        return at_least_one_strictly_greater
    

    @staticmethod
    def cholesky_decomposition(matrix, sig_figs):
        n = len(matrix)
        lower = [[0.0] * n for _ in range(n)]
        steps = [] 
        
        for i in range(n):
            for j in range(i + 1):
                sum_val = 0.0
                if j == i: 
                    for k in range(j): sum_val += lower[j][k]**2
                    val = matrix[j][j] - sum_val
                    if val <= 0: raise ValueError("Matrix is not Positive Definite!")
                    lower[j][j] = math.sqrt(val)
                    steps.append({"type": "diag", "i": i, "j": j, 
                        "formula": f"L_{{{i}{i}}} = \\sqrt{{A_{{{i}{i}}} - \\sum L_{{{i}k}}^2}}",
                        "res": MatrixSolver._fmt(lower[j][j], sig_figs)})
                else: 
                    for k in range(j): sum_val += lower[i][k] * lower[j][k]
                    lower[i][j] = (matrix[i][j] - sum_val) / lower[j][j]
                    steps.append({"type": "off", "i": i, "j": j, 
                        "formula": f"L_{{{i}{j}}} = ...", 
                        "res": MatrixSolver._fmt(lower[i][j], sig_figs)})
        return lower, steps
    

    # Jacobi with relaxation for improved convergence (called weighted Jacobi)
    # the parameter n can be removed as we can get the no of vars from the npArray
    @staticmethod
    def Jacobi_noNorm(A,b,x,maxIterations,ErrorTolerance,relax=1 , significantFigs = 7 , rounding = True) :
        # Setting up signifcant figs and rounding/chopping
        intializeContext(significantFigs,rounding)
        #Copying the arrays to avoid modifying original arrays
        A = A.copy()
        b = b.copy()
        x = x.copy()
        n = len(b)
        #converting floats to decimals 
        A = toDecimal(A)
        b = toDecimal(b)
        x = toDecimal(x)
        relax = decimal.Decimal(str(relax))
        #Array to hold new values
        xNew = np.zeros(n,dtype=object)
        #List to hold iteration details
        steps = []

        if(MatrixSolver.isDiagonalyDominant(A) ) :
            steps.append('The matrix is diagonaly dominant')
        else :
            steps.append('The matrix is not diagonaly dominant')
        var_steps = []
        # Calculating first iteration before applying relaxation    
        for i in range(n) :
            computation_terms = [f"{b[i]}"]  # start with b_i
            sum = b[i]
            for j in range(n) :
                if(i==j) :
                    continue
                sum -= A[i][j] * x[j]
                computation_terms.append(f" - ({A[i][j]} * {x[j]})")
            formula_str = "".join(computation_terms)
            xNew[i] = sum/A[i][i]
            var_steps.append(f"x{i+1} = ({formula_str} / {A[i][i]}) = {xNew[i]}")

        x = xNew.copy()
        # Storing details of first iteration    
        details = {
                'type': 'iter',
                'k' : 1,
                'x_vec' : toFloats(x),
                'steps' : var_steps,
                'error' : '_'
            }
        steps.append(details)
        iteration = 2
        # Loop until convergence or max iterations reached 
        while (True) :
            belowTolerance = True
            maxError = 0
            var_steps = []
            for i in range(n) :
                oldX = x[i]
                sum = b[i]
                computation_terms = [f"{b[i]}"]  # for formula string
                for j in range(n) :
                    if(i==j) :
                        continue
                    sum -= A[i][j] * x[j]
                    computation_terms.append(f" - ({A[i][j]} * {x[j]})")
                xNew[i] = relax*sum/A[i][i] + (1-relax)*oldX
                if (xNew[i] != 0) :
                    estimatedError = abs((xNew[i]-oldX)/xNew[i]) * 100
                    maxError = max(maxError, estimatedError)
                    if(estimatedError > ErrorTolerance):
                        belowTolerance = False
                full_formula = (
                f"x{i+1} = {relax}*(({''.join(computation_terms)}) / {A[i][i]})"
                f" + (1-{relax})*{oldX} = {xNew[i]}"
            )        
                var_steps.append(full_formula)     
            details = {
                'type' : 'iter',
                'k':iteration,
                'x_vec' : toFloats(xNew),
                'steps': var_steps,
                'error' : float(maxError)
            } 
            steps.append(details)
            
            
            iteration+=1

            if belowTolerance:
                break;
        
            if iteration > maxIterations:
                steps.append(f"Reached max iterations ({maxIterations}) without full convergence, final error = {float(maxError)}")
                break

        return toFloats(xNew) , steps

    @staticmethod
    def GaussSeidel_noNorm(A: np.array, b: np.array, x , maxIterations,ErrorTolerance,relax=1, significantFigs = 7 , rounding = True) :
        # Setting up signifcant figs and rounding/chopping
        intializeContext(significantFigs,rounding)
        
        #converting floats to decimals 
        A = toDecimal(A)
        b = toDecimal(b)
        if x is not None:
            x = toDecimal(x)
        n = len(b)
        relax = decimal.Decimal(str(relax))
        
        #List to hold iteration details
        steps = []

        #Check for diagonaly dominant matrix
        if(MatrixSolver.isDiagonalyDominant(A) ) :
            steps.append('The matrix is diagonaly dominant')
        else :
            steps.append('The matrix is not diagonaly dominant')

        
        # Calculating first iteration before applying relaxation    
        var_steps = []
        for i in range(n) :
            computation_terms = [f"{b[i]}"]  # start with b_i
            sum_val = b[i]
            for j in range(n) :
                if(i==j) :
                    continue
                sum_val -= A[i][j] * x[j]
                computation_terms.append(f" - ({A[i][j]} * {x[j]})")
            formula_str = "".join(computation_terms)
            x[i] = sum_val / A[i][i]
            var_steps.append(f"x{i+1} = ({formula_str} / {A[i][i]}) = {x[i]}")
        
        details = {
                'type': 'iter',
                'k' : 1,
                'steps': var_steps,
                'x_vec' : toFloats(x),
                'error' : '_'
            }     
        
        steps.append(details)
        iteration = 2
        # Loop until convergence or max iterations reached 
        while (True) :
            belowTolerance = True
            maxError = 0
            var_steps = []
            for i in range(n) :
                oldX = x[i]
                sum_val = b[i]
                computation_terms = [f"{b[i]}"]  # for formula string
                for j in range(n) :
                    if(i==j) :
                        continue
                    sum_val -= A[i][j] * x[j]
                    computation_terms.append(f" - ({A[i][j]} * {x[j]})")
                formula_str = "".join(computation_terms)
                # Relaxation formula
                x[i] = relax*sum_val/A[i][i] + (1-relax)*oldX

                if (x[i] != 0) :
                    estimatedError = abs(float((x[i]-oldX)/x[i])) * 100
                    estimatedError = float(estimatedError)
                    if(estimatedError > ErrorTolerance):
                        belowTolerance = False
                    maxError = max(maxError, estimatedError)
                full_formula = (
                f"x{i+1} = {relax}*(({''.join(computation_terms)}) / {A[i][i]})"
                f" + (1-{relax})*{oldX} = {x[i]}"
            )
                var_steps.append(full_formula)     
            details = {
                'type' : 'iter',
                'k':iteration,
                'x_vec' : toFloats(x),
                'steps': var_steps,
                'error' : MatrixSolver._fmt(maxError, significantFigs)
            }        

            steps.append(details)

            iteration+=1
            
            if belowTolerance:
                break;
        
            if iteration > maxIterations:
                steps.append(f"Reached max iterations ({maxIterations}) without full convergence, final error = {float(maxError)}")
                break

        # x = toFloats(x).tolist()
        
        return toFloats(x), steps
    

    @staticmethod
    def LU_decomposition(A, b=[], scaling=False, sig_figs=4):
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
                        "type": "calc_l",
                        "i":"i",
                        "j":k,
                        "formula":f"L_{{{i}{k}}} = A_{{{i}{k}}} / A_{{{k}{k}}}",
                        "res": MatrixSolver._fmt(factor,sig_figs)
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


    @staticmethod
    def solve_from_LU(A, b, scaling, sig_figs=4):
        L, U, P, er, steps = MatrixSolver.LU_decomposition(A, b, scaling, sig_figs)
        if er == -1:
            return "No solution"  
        elif er == 0:
            return "Infinite solutions"
        
        n = len(A)
        # Apply row permutation to b
        b_permuted = [b[P[i]] for i in range(n)]    # Solve Ly = Pb
        steps.append(f"Permuted vector b according to P: {MatrixSolver._fmt_vec(b_permuted, sig_figs)}")

        y = forward_substitution(L, b_permuted, n)
        steps.append(f"Forward substitution result (y) = {MatrixSolver._fmt_vec(y, sig_figs)}")
        # Solve Ux = y
        x = back_substitution(U, y, n)
        return x, L, U, steps




    
    @staticmethod
    def solve_lu(A, b, sig_figs):
        # Implementation of LU (Doolittle)
        n = len(A)
        L = [[0.0]*n for _ in range(n)]
        U = [[0.0]*n for _ in range(n)]
        steps = []
        for i in range(n): L[i][i] = 1.0

        for i in range(n):
            for k in range(i, n):
                s = sum(L[i][j] * U[j][k] for j in range(i))
                U[i][k] = A[i][k] - s
                steps.append({"type": "calc_u", "i": i, "j": k, "formula": "...", "res": MatrixSolver._fmt(U[i][k], sig_figs)})
            for k in range(i+1, n):
                s = sum(L[k][j] * U[j][i] for j in range(i))
                L[k][i] = (A[k][i] - s) / U[i][i]
                steps.append({"type": "calc_l", "i": k, "j": i, "formula": "...", "res": MatrixSolver._fmt(L[k][i], sig_figs)})

        # Forward/Backward Subs
        y = [0.0]*n
        for i in range(n):
            y[i] = (b[i] - sum(L[i][j]*y[j] for j in range(i))) / L[i][i]
        x = [0.0]*n
        for i in range(n-1, -1, -1):
            x[i] = (y[i] - sum(U[i][j]*x[j] for j in range(i+1, n))) / U[i][i]
            
        return x, L, U, steps

    @staticmethod
    def _fmt_vec(vec, sig_figs=4):
        return "[" + ", ".join([MatrixSolver._fmt(v, sig_figs) for v in vec]) + "]"
    
    # --- Formatting Helper ---
    @staticmethod
    def _fmt(val, sig_figs=4):
        """Formats a number to the specified significant figures."""
        if val == 0: return "0"
        try:
            return f"{val:.{sig_figs}g}"
        except:
            return str(val)
    
    @staticmethod
    def _log_matrix(A, b, sig_figs):
        """Helper to format the current matrix state as a clean HTML block."""
        rows = len(A)
        # Container with dark background and monospaced font
        html = "<div class='mt-2 mb-4 inline-block bg-slate-900 rounded p-3 font-mono text-xs border border-slate-700 shadow-inner'>"
        
        for i in range(rows):
            # Format row values with padding for alignment
            row_str = "  ".join([f"{MatrixSolver._fmt(val, sig_figs):>8}" for val in A[i]])
            
            # Add the augmented vector 'b' if it exists
            b_str = f"  |  {MatrixSolver._fmt(b[i], sig_figs)}" if b is not None else ""
            
            # Combine into a single line
            html += f"<div class='whitespace-pre hover:bg-slate-800/50 px-1 rounded'>{row_str}{b_str}</div>"
        
        html += "</div>"
        return html
