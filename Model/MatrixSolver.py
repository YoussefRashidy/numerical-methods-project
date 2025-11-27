import math

from src.utils.precision_rounding import toDecimal, toFloats, intializeContext
import decimal
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

        # Calculating first iteration before applying relaxation    
        for i in range(n) :
            sum = b[i]
            for j in range(n) :
                if(i==j) :
                    continue
                sum -= A[i][j] * x[j]
            xNew[i] = sum/A[i][i]
        x = xNew.copy()
        # Storing details of first iteration    
        details = {
                'type': 'iter',
                'k' : 1,
                'x_vec' : toFloats(x),
                'error' : '_'
            }
        steps.append(details)
        iteration = 2
        # Loop until convergence or max iterations reached 
        while (True) :
            belowTolerance = True
            maxError = decimal.Decimal('0')
            for i in range(n) :
                oldX = x[i]
                sum = b[i]
                for j in range(n) :
                    if(i==j) :
                        continue
                    sum -= A[i][j] * x[j]
                xNew[i] = relax*sum/A[i][i] + (1-relax)*oldX
                if (belowTolerance and xNew[i] != 0) :
                    estimatedError = abs((xNew[i]-oldX)/xNew[i]) * 100
                    maxError = max(maxError, estimatedError)
                    if(estimatedError > ErrorTolerance):
                        belowTolerance = False
            details = {
                'type' : 'iter',
                'k':iteration,
                'x_vec' : toFloats(x),
                'error' : float(maxError)
            }      
            steps.append(details)
            iteration+=1
            x = xNew.copy()
            
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
        for i in range(n) :
            sum_val = b[i]
            for j in range(n) :
                if(i==j) :
                    continue
                sum_val -= A[i][j] * x[j]

            x[i] = sum_val / A[i][i]
        
        details = {
                'type': 'iter',
                'k' : 1,
                'x_vec' : toFloats(x),
                'error' : '_'
            }     
        
        steps.append(details)
        iteration = 2
        # Loop until convergence or max iterations reached 
        while (True) :
            belowTolerance = True
            maxError = 0
            for i in range(n) :
                oldX = x[i]
                sum_val = b[i]
                for j in range(n) :
                    if(i==j) :
                        continue
                    sum_val -= A[i][j] * x[j]

                # Relaxation formula
                x[i] = relax*sum_val/A[i][i] + (1-relax)*oldX

                if (x[i] != 0) :
                    estimatedError = abs(float((x[i]-oldX)/x[i])) * 100
                    estimatedError = float(estimatedError)
                    if(estimatedError > ErrorTolerance):
                        belowTolerance = False
                    maxError = max(maxError, estimatedError)
            details = {
                'type' : 'iter',
                'k':iteration,
                'x_vec' : toFloats(x),
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
    def _fmt_vec(vec, sig_figs):
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
    
    
