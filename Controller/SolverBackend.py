import math
from Model.MatrixSolver import MatrixSolver
class SolverBackend:
    """
    The 'Service' Class. 
    Acts as a library of numerical algorithms.
    """

    # --- The Dispatcher (The MUX) ---
    def solve(self, method, A, b, steps_log=False, tol=None, max_iter=None, sig_figs=None):
        if method == "Cholesky Decomposition":
            return self.solve_cholesky(A, b)
        elif method == "Gaussian Elimination":
            return self.solve_gaussian(A, b)
        elif method == "Jacobi Iteration":
            if tol is None or max_iter is None:
                raise ValueError("Tolerance and Max Iterations required for Jacobi.")
            return self.solve_jacobi(A, b, tol, max_iter)
        else:
            raise ValueError(f"Method '{method}' is not implemented yet.")

    # --- Formatting Helper (Larger Font) ---
    def _matrix_to_html(self, matrix, highlight_cell=None):
        """
        Converts a 2D list into an HTML table.
        """
        html = '<table border="1" cellspacing="0" cellpadding="5" style="border-color: #bdc3c7; border-collapse: collapse; margin: 10px;">'
        
        for r, row in enumerate(matrix):
            html += "<tr>"
            for c, val in enumerate(row):
                # CHANGED: font-size from 14px to 18px
                style = "padding: 8px 15px; text-align: center; color: #ecf0f1; font-size: 18px; border: 1px solid #7f8c8d;"
                
                val_str = f"{val:.4f}".rstrip('0').rstrip('.') if val != 0 else "0"
                html += f'<td style="{style}">{val_str}</td>'
            html += "</tr>"
        
        html += "</table>"
        return html    

    def solve_cholesky(self, A, b):
        L, steps = self.cholesky_decomposition(A) 
        L_T = [[L[j][i] for j in range(len(L))] for i in range(len(L))]
        
        steps.append("<h3>Step 2: Forward Substitution (Ly = b)</h3>")
        y = self.forward_substitution(L, b)
        y_html = self._matrix_to_html([y]) 
        steps.append(f"<p style='font-size: 16px;'>Intermediate Vector y:</p>{y_html}")
        
        steps.append("<h3>Step 3: Backward Substitution (L<sup>T</sup>x = y)</h3>")
        x = self.backward_substitution(L_T, y)
        
        return x, L, steps

    # --- Helper Methods ---
    def forward_substitution(self, L, b):
        n = len(b)
        y = [0.0] * n
        for i in range(n):
            sum_val = sum(L[i][j] * y[j] for j in range(i))
            y[i] = (b[i] - sum_val) / L[i][i]
        return y

    def backward_substitution(self, U, y):
        n = len(y)
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            sum_val = sum(U[i][j] * x[j] for j in range(i + 1, n))
            x[i] = (y[i] - sum_val) / U[i][i]
        return x

    def solve_gaussian(self, A, b):
        raise NotImplementedError("Gaussian Logic coming soon...")

    def solve_jacobi(self, A, b, tol, max_iter):
        raise NotImplementedError("Jacobi Logic coming soon...")