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

        if x_root_old is not None and abs(x_root - x_root_old) < tol :
            break 

        x_root_old = x_root

    return (x_root, relativeError , len(iteration_details) , iteration_details)

if "__name__" == "__main__":
    print("hi")
        