from src.utils.precision_rounding import intializeContext
import decimal
from src.utils.roots_utils import parse_exp , parse_derivative

def fixed_point_iteration(x0,func_expr,tol=1e-7 ,max_iter=100 , significantFigs = 14 , rounding = True):
    intializeContext(significantFigs,rounding=rounding)
    x_root = decimal.Decimal(x0)
    relativeError = "---"
    g = parse_exp(func_expr)
    iteration_details = []
    iter = 0 
    for i in range(max_iter) :
        x_root_old = x_root
        x_root = g(x_root_old)
        if x_root != 0 and i != 0 :
            relativeError = abs((x_root-x_root_old)/ x_root) * 100
            
        if func_expr != "" :
            expr_eval_str = f"{func_expr.replace('x', str(x_root_old))} = {x_root}"
        else:
            expr_eval_str = str(x_root)    
            
        details = {"iter" : i+1,
                   "x_new" : str(x_root) ,
                   "x_old": str(x_root_old),
                   "error" : str(relativeError) , 
                   "Evaluation" : expr_eval_str
                   }
        
        iteration_details.append(details)
        if abs(x_root-x_root_old) <= tol :
            break 
    return (x_root, relativeError, len(iteration_details), iteration_details)
        