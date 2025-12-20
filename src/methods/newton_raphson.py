from src.utils.precision_rounding import intializeContext
import decimal
import sympy as sp
from src.utils.roots_utils import parse_exp , parse_derivative
def NewtonRaphson(x0,func_expr,tol=1e-7 ,max_iter=100 , significantFigs = 14 , rounding = True):
    intializeContext(significantFigs,rounding=rounding)

    x_root = decimal.Decimal(str(x0))

    # relativeError = decimal.Decimal("Infinity")

    func = parse_exp(func_expr)
    derivative = parse_derivative(func_expr)
    
    deriv_exp = str(sp.diff(sp.sympify(func_expr)))

    relativeError = "---"
    x_root_old = None

    iteration_details = []

    for i in range(1, max_iter+1) :

        f_val = func(x_root)
        f_prime_val = derivative(x_root)

        if f_prime_val == 0:
            raise RuntimeError("Derivative is zero!")

        
        x_root = x_root - (f_val / f_prime_val)

        if i == 1:
            relativeError = "---"
        else:
            if x_root == 0:
                relativeError = "Undefined"
            else:
                relativeError = str((abs(x_root-x_root_old) / x_root)*100)

            
        if func_expr != "" :
            expr_eval_str = f"{x_root_old}-(({func_expr.replace('x', str(x_root_old))})/ ({deriv_exp.replace('x', str(x_root_old))})) = {x_root}"
        else:
            expr_eval_str = str(x_root)    
            
        detials = {
                    "iter" : i,
                    "x_old" : str(x_root_old),
                    "x_new" : str(x_root) ,
                    "error" : relativeError ,
                    "Evaluation" : expr_eval_str
                    }
        
        iteration_details.append(detials)

        if x_root_old is not None and ((abs(x_root-x_root_old) / x_root)*100) < tol :
            break 

        x_root_old = x_root

    return (x_root, relativeError , len(iteration_details) , iteration_details)


def ModifiedNewtonRaphson(x0,func_expr,tol=1e-7 ,max_iter=100 , significantFigs = 14 , rounding = True):
    intializeContext(significantFigs,rounding=rounding)

    x_root = decimal.Decimal(str(x0))

    # relativeError = decimal.Decimal("Infinity")

    func = parse_exp(func_expr)
    derivative = parse_derivative(func_expr)


    # Build derivative expressions using sympy
    deriv_exp = str(sp.diff(sp.sympify(func_expr))) 
    d2_expr = str(sp.diff(sp.sympify(deriv_exp)))     
    second_derivative = parse_exp(d2_expr)  
    

    relativeError = "---"
    x_root_old = None

    iteration_details = []

    for i in range(1, max_iter+1) :

        f_val = func(x_root)
        f_prime_val = derivative(x_root)
        f_double_prime_val = second_derivative(x_root)

        if f_prime_val == 0:
            raise RuntimeError("Derivative is zero!")

        if f_double_prime_val == 0:
            raise RuntimeError("Second derivative is zero!")

        denominator = f_prime_val**2 - f_val * f_double_prime_val
        
        x_root_old = x_root

        x_root = x_root - (f_val*f_prime_val)/denominator

        if i == 1:
            relativeError = "---"
        else:
            if x_root == 0:
                relativeError = "Undefined"
            else:
                relativeError = str((abs((x_root-x_root_old) / x_root))*100)

            
        if func_expr != "" :
            expr_eval_str = f"{x_root_old}-(({func_expr.replace('x', str(x_root_old))})/ ({deriv_exp.replace('x', str(x_root_old))})) = {x_root}"
        else:
            expr_eval_str = str(x_root)    
            
        detials = {
                    "iter" : i,
                    "x_old" : str(x_root_old),
                    "x_new" : str(x_root) ,
                    "error" : relativeError ,
                    "Evaluation" : expr_eval_str
                    }
        
        iteration_details.append(detials)

        if x_root_old is not None and abs(((x_root - x_root_old)/x_root)*100) < tol :
            break 

    return (x_root, relativeError , len(iteration_details) , iteration_details)

def ModifiedNewtonRaphsonmmm(x0, func_expr, tol=1e-7, max_iter=100,
                          significantFigs=14, rounding=True):
    """
    Modified Newton-Raphson method with second derivative correction.
    Uses the update formula:
    
        x_{n+1} = x_n - ( f(x_n) * f'(x_n) ) / ( (f'(x_n))^2 - f(x_n) * f''(x_n) )
    
    If the denominator becomes too small, the algorithm falls back
    to a standard Newton step: x_{n+1} = x_n - f(x_n) / f'(x_n).

    Returns:
        x_root        : final computed root
        relativeError : absolute error of last iteration
        num_iters     : number of iterations performed
        steps         : list of dictionaries with iteration history
    """

    # Set decimal precision and rounding mode
    intializeContext(significantFigs, rounding=rounding)

    # Convert initial guess to Decimal
    x_root = decimal.Decimal(x0)
    relativeError = decimal.Decimal("Infinity")

    # Parse function and derivatives
    func = parse_exp(func_expr)                   
    derivative = parse_derivative(func_expr)      

    # Build derivative expressions using sympy
    deriv_exp = str(sp.diff(sp.sympify(func_expr))) 
    d2_expr = str(sp.diff(sp.sympify(deriv_exp)))     
    second_derivative = parse_exp(d2_expr)          

    steps = []

    for i in range(max_iter):
        x_old = x_root

        # Evaluate f(x), f'(x), f''(x)
        fx = func(x_old)
        fpx = derivative(x_old)
        fppx = second_derivative(x_old)

        # Very small threshold to detect division issues
        eps = decimal.Decimal("1e-30")
        denominator = fpx**2 - fx * fppx

        # If denominator nearly zero → fallback to standard Newton step
        if abs(denominator) < eps:
            # Cannot proceed if derivative is zero
            if fpx == 0:
                break
            x_new = x_old - fx / fpx
        else:
            # Modified Newton–Raphson update
            x_new = x_old - (fx * fpx) / denominator

        # Compute absolute error
        if x_new != 0:
            relativeError = abs((x_new - x_old) / x_new)
        else:
            relativeError = abs(x_old)

        # Save iteration information
        steps.append({
            "iter": i + 1,
            "x_old": x_old,
            "x_new": x_new,
            "error": relativeError
        })

        # Update current root estimate
        x_root = x_new

        # Stop if tolerance is satisfied
        if relativeError <= tol:
            break

    # Return results in the requested format
    return x_root, relativeError, len(steps), steps
        