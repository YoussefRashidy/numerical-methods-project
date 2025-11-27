import math
import time
from Model.MatrixSolver import MatrixSolver
import numpy as np

class SolverBackend:
    """
    The 'Service' Class. 
    Acts as a library of numerical algorithms.
    """

    def solve(self, method, A, b, steps_log=False, tol=None, max_iter=None, sig_figs=4, x_init=None):
        """
        Routes the request to the correct algorithm.
        Returns: x, L, U, steps, execution_time
        """
        # Convert to numpy arrays
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        if x_init is not None:
            x_init = np.array(x_init, dtype=float)
        
        start_time = time.perf_counter()


        
        try:
            if method == "Cholesky Decomposition":
                x, L, steps = self.solve_cholesky(A, b, sig_figs)
                U = None
            
            elif method == "Gaussian Elimination":
                x, U, steps = self.solve_gaussian(A, b, sig_figs) # Partial implementation
                L = None
                
            elif method == "LU Decomposition":
                x, L, U, steps = self.solve_lu(A, b, sig_figs)
                
            elif method == "Jacobi Iteration":
                if tol is None or max_iter is None:
                    raise ValueError("Tolerance and Max Iterations required for Jacobi.")
                x, steps = MatrixSolver.Jacobi_noNorm(A, b, x_init, max_iter, tol, significantFigs=sig_figs)
                L, U = None, None
                
            elif method == "Gauss-Seidel Iteration":
                if tol is None or max_iter is None:
                    raise ValueError("Tolerance and Max Iterations required for Gauss-Seidel.")
                x,steps = MatrixSolver.GaussSeidel_noNorm(A, b, x_init, max_iter, tol, significantFigs=sig_figs)
                #  x, steps = self.solve_iterative(A, b, x_init, tol, max_iter, sig_figs, method="Gauss-Seidel")
                L, U = None, None
                 
            else:
                raise ValueError(f"Method '{method}' is not implemented yet.")
            
            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000 # Convert to ms
            
            return x, L, U, steps, execution_time

        except Exception as e:
            raise e

    # --- Direct Methods ---
    def solve_cholesky(self, A, b, sig_figs):
        
        L, steps = MatrixSolver.cholesky_decomposition(A, sig_figs)
        
        n = len(b)
        y = [0.0] * n
        for i in range(n):
            sum_val = sum(L[i][j] * y[j] for j in range(i))
            y[i] = (b[i] - sum_val) / L[i][i]

        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            sum_val = sum(L[j][i] * x[j] for j in range(i + 1, n))
            x[i] = (y[i] - sum_val) / L[i][i]
            
        return x, L, steps

    def solve_lu(self, A, b, sig_figs):
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
                steps.append({"type": "calc_u", "i": i, "j": k, "formula": "...", "res": self._fmt(U[i][k], sig_figs)})
            for k in range(i+1, n):
                s = sum(L[k][j] * U[j][i] for j in range(i))
                L[k][i] = (A[k][i] - s) / U[i][i]
                steps.append({"type": "calc_l", "i": k, "j": i, "formula": "...", "res": self._fmt(L[k][i], sig_figs)})

        # Forward/Backward Subs
        y = [0.0]*n
        for i in range(n):
            y[i] = (b[i] - sum(L[i][j]*y[j] for j in range(i))) / L[i][i]
        x = [0.0]*n
        for i in range(n-1, -1, -1):
            x[i] = (y[i] - sum(U[i][j]*x[j] for j in range(i+1, n))) / U[i][i]
            
        return x, L, U, steps

    def solve_gaussian(self, A, b, sig_figs):
         raise NotImplementedError("Gaussian Elimination not fully implemented yet.")