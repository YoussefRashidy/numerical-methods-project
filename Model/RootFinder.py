from src.methods import Bisection, false_position

class RootFinder:
    """
    A static collection of Root Finding Algorithms.
    Acts as a facade/dispatcher for the specific method implementations.
    """

    @staticmethod
    def solve(method, equation, tol=1e-5, max_iter=50, sig_figs=5, xl=None, xu=None, x0=None, x1=None, g_equation=None):
        """
        Dispatches the request to the specific root finding algorithm.
        
        Parameters:
        - method: String name of the method (e.g., "Bisection")
        - equation: String function f(x)
        - tol: Tolerance (epsilon)
        - max_iter: Maximum iterations
        - sig_figs: Significant figures for precision
        - xl, xu: Bracketing limits (for Bisection/False-Position)
        - x0: Initial guess (for Newton/Fixed-Point)
        - x1: Second guess (for Secant)
        - g_equation: g(x) for Fixed Point method
        
        Returns:
        - root (float)
        - ea (relative error %)
        - iter_count (int)
        - steps (list of dicts/strings)
        - time (float) - execution time is usually measured by the caller (SolverBackend)
        """
        
        # Validate inputs based on method
        if method == "Bisection":
            if xl is None or xu is None:
                raise ValueError("Bisection requires lower (xl) and upper (xu) bounds.")
            # return bisection(equation, xl, xu, tol, max_iter, sig_figs)
            return Bisection.Bisection(x1=xl, x2=xu, func_expr=equation,tol= tol,max_iter= max_iter,significantFigs= sig_figs)

        elif method == "False-Position":
            if xl is None or xu is None:
                raise ValueError("False-Position requires lower (xl) and upper (xu) bounds.")
            # return false_position(equation, xl, xu, tol, max_iter, sig_figs)
            return None, None, None, ["False-Position not implemented yet"]

        elif method == "Fixed Point":
            if x0 is None:
                raise ValueError("Fixed Point requires an initial guess (x0).")
            # Note: Fixed point might need g(x) passed explicitly or derived
            # return fixed_point(equation, x0, tol, max_iter, sig_figs)
            return None, None, None, ["Fixed Point not implemented yet"]

        elif method == "Newton-Raphson":
            if x0 is None:
                raise ValueError("Newton-Raphson requires an initial guess (x0).")
            # return newton(equation, x0, tol, max_iter, sig_figs)
            return None, None, None, ["Newton-Raphson not implemented yet"]

        elif method == "Modified Newton":
            if x0 is None:
                raise ValueError("Modified Newton requires an initial guess (x0).")
            # return modified_newton(equation, x0, tol, max_iter, sig_figs)
            return None, None, None, ["Modified Newton not implemented yet"]

        elif method == "Secant":
            # Secant needs two guesses x0 (x_{i-1}) and x1 (x_i)
            if x0 is None or x1 is None:
                raise ValueError("Secant method requires two initial guesses (x0, x1).")
            # return secant(equation, x0, x1, tol, max_iter, sig_figs)
            return None, None, None, ["Secant not implemented yet"]

        else:
            raise ValueError(f"Root finding method '{method}' is not supported.")