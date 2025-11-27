import math
from src.methods.gauss_seidel import GaussSeidel_noNorm
from src.methods.gauss_seidel import isDiagonalyDominant
from src.methods.Jacobi import Jacobi_noNorm
class MatrixSolver:
    def cholesky_decomposition(self, matrix):
        """Decomposes symmetric, positive-definite matrix A into L * L.T"""
        n = len(matrix)
        lower = [[0.0] * n for _ in range(n)]
        steps = [] 

        steps.append("<h3>Step 1: Cholesky Decomposition</h3>")
        steps.append(f"<p style='font-size: 16px;'>Starting Matrix A:</p>{self._matrix_to_html(matrix)}")

        for i in range(n):
            for j in range(i + 1):
                sum_val = 0.0
                
                if j == i: # Diagonal
                    steps.append(f"<hr><b>Calculating Diagonal L<sub>{i},{i}</sub>:</b>") 
                    for k in range(j):
                        sum_val += lower[j][k]**2
                    
                    temp = matrix[j][j] - sum_val
                    
                    formula = f"L<sub>{i},{i}</sub> = &radic;(A<sub>{i},{i}</sub> - &Sigma; L<sub>{i},k</sub><sup>2</sup>)"
                    calc = f"&radic;({matrix[j][j]} - {sum_val:.4f})"
                    # CHANGED: Added font-size: 16px
                    steps.append(f"<p style='color:#bdc3c7; font-size: 16px;'><i>Formula:</i> {formula}<br><i>Calc:</i> {calc}</p>")

                    if temp <= 0:
                        raise ValueError("Matrix is not Positive Definite (Non-positive pivot encountered)!")
                    
                    lower[j][j] = math.sqrt(temp)
                    steps.append(f"<b>Result:</b> L<sub>{i},{i}</sub> = {lower[j][j]:.4f}") 
                
                else: # Off-Diagonal
                    steps.append(f"<hr><b>Calculating Off-Diagonal L<sub>{i},{j}</sub>:</b>") 
                    for k in range(j):
                        sum_val += lower[i][k] * lower[j][k]
                    
                    formula = f"L<sub>{i},{j}</sub> = (A<sub>{i},{j}</sub> - &Sigma; L<sub>{i},k</sub> &middot; L<sub>{j},k</sub>) / L<sub>{j},{j}</sub>"
                    calc = f"({matrix[i][j]} - {sum_val:.4f}) / {lower[j][j]:.4f}"
                    # CHANGED: Added font-size: 16px
                    steps.append(f"<p style='color:#bdc3c7; font-size: 16px;'><i>Formula:</i> {formula}<br><i>Calc:</i> {calc}</p>")
                    
                    lower[i][j] = (matrix[i][j] - sum_val) / lower[j][j]
                    steps.append(f"<b>Result:</b> L<sub>{i},{j}</sub> = {lower[i][j]:.4f}") 
                
                steps.append(f"<br>Current L Matrix:{self._matrix_to_html(lower, highlight_cell=(i,j))}")

        return lower, steps
      # --- Added Methods ---
    def gauss_seidel(self, A, b, x0, max_iter, tol, relax=1 , significantFigs=7 , rounding=True):
        """Wrapper for Gauss–Seidel method."""
        return GaussSeidel_noNorm(A, b,len(b) ,x0, max_iter, tol, relax, significantFigs, rounding)
    
    def jacobi(self, A, b, x0, max_iter, tol , relax=1 , significantFigs=7 , rounding=True):
        """Wrapper for Jacobi method."""
        return Jacobi_noNorm(A, b,len(b), x0, max_iter, tol,relax=relax , significantFigs= significantFigs , rounding = rounding)
    
    def is_diagonally_dominant(self, A):
        """Check if matrix A is diagonally dominant."""
        return isDiagonalyDominant(A)