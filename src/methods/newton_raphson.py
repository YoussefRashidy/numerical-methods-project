from src.utils.precision_rounding import intializeContext
import decimal
from src.utils.roots_utils import parse_exp , parse_derivative
def NewtonRaphson(x0,func_expr,tol=1e-7 ,max_iter=100 , significantFigs = 14 , rounding = True):
    intializeContext(significantFigs,rounding=rounding)
    x_root = decimal.Decimal(x0)
    relativeError = decimal.Decimal("Infinity")
    func = parse_exp(func_expr)
    derivative = parse_derivative(func_expr)
    
    deriv_exp = str(sp.diff(sp.sympify(func_expr)))
    iteration_details = []
    iter = 0 
    for i in range(max_iter) :
        x_root_old = x_root
        x_root = x_root_old - (func(x_root_old) / derivative(x_root_old))
        if x_root != 0 and i != 0 :
            relativeError = abs(x_root-x_root_old)
            
        if func_expr != "" :
            expr_eval_str = f"{x_root_old}-(({func_expr.replace('x', str(x_root_old))})/ ({deriv_exp.replace('x', str(x_root_old))})) = {x_root}"
        else:
            expr_eval_str = str(x_root)    
            
        detials = {"iteration" : i+1,"x_root_old" : x_root_old, "x_root" : x_root , "Relative Error" : relativeError , "Evaluation" : expr_eval_str}
        
        iteration_details.append(detials)
        if relativeError <= tol :
            break 
    return x_root, iteration_details
        